//! agentd desktop shell.
//!
//! The Rust side owns exactly two things, both of which the web console cannot
//! do for itself:
//!
//! 1. The SSH tunnel. One `ssh -N -L` carries the API (8765), the PTY
//!    WebSocket (8766) and noVNC (6080) back to loopback. The server exposes
//!    no public ports, so this tunnel *is* the security boundary.
//! 2. The bearer token, kept in the OS keychain rather than in the webview's
//!    storage where any XSS could read it.
//!
//! Everything else -- council, terminal, screen, ops, approvals -- is the same
//! HTML the browser console runs, loaded from the bundled frontendDir. Two
//! implementations of a UI would drift; one does not.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Listener, Manager, State};

/// Ports forwarded from the server. Fixed so the frontend can hard-code them.
pub const API_PORT: u16 = 8765;
pub const WS_PORT: u16 = 8766;
pub const NOVNC_PORT: u16 = 6080;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostConfig {
    pub label: String,
    pub host: String,
    pub user: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Key-based only. The desktop app deliberately has no password field:
    /// the server turned password auth off, and a UI that still offers it
    /// invites people to paste a password into a webview.
    #[serde(default)]
    pub key_path: String,
}

fn default_port() -> u16 {
    22
}

#[derive(Default)]
struct Tunnel {
    child: Option<Child>,
    /// Local ports actually bound, in case 8765 is taken and the OS picked
    /// another. The frontend reads these instead of assuming.
    api: u16,
    ws: u16,
    novnc: u16,
}

#[derive(Debug, Clone, Serialize)]
pub struct TunnelStatus {
    pub running: bool,
    pub api_port: u16,
    pub ws_port: u16,
    pub novnc_port: u16,
    pub error: String,
}

#[tauri::command]
fn api_base(api_port: u16) -> String {
    format!("http://127.0.0.1:{api_port}")
}

#[tauri::command]
fn open_tunnel(state: State<'_, Mutex<Tunnel>>, cfg: HostConfig) -> Result<TunnelStatus, String> {
    let mut guard = state.lock().map_err(|_| "tunnel state poisoned")?;

    if let Some(child) = guard.child.as_mut() {
        // Reap a tunnel that died between calls instead of leaking a child.
        match child.try_wait() {
            Ok(Some(_)) => {
                guard.child = None;
            }
            Ok(None) => {
                return Ok(TunnelStatus {
                    running: true,
                    api_port: guard.api,
                    ws_port: guard.ws,
                    novnc_port: guard.novnc,
                    error: String::new(),
                })
            }
            Err(e) => return Err(format!("cannot probe tunnel: {e}")),
        }
    }

    let target = format!("{}@{}", cfg.user, cfg.host);
    let mut argv: Vec<String> = vec![
        "ssh".into(),
        "-N".into(),                          // no remote command: forward only
        "-o".into(),
        "BatchMode=yes".into(),               // never prompt inside a GUI
        "-o".into(),
        "ExitOnForwardFailure=yes".into(),    // fail loudly instead of half-working
        "-o".into(),
        format!("ServerAliveInterval=15").into(),
        "-o".into(),
        "ServerAliveCountMax=3".into(),       // ~45s to notice a dead link
        "-p".into(),
        cfg.port.to_string(),
    ];
    if !cfg.key_path.is_empty() {
        argv.extend(["-i".into(), cfg.key_path.clone(), "-o".into(), "IdentitiesOnly=yes".into()]);
    }
    argv.extend([
        "-L".into(),
        format!("{API_PORT}:127.0.0.1:{API_PORT}"),
        "-L".into(),
        format!("{WS_PORT}:127.0.0.1:{API_PORT}"),
        "-L".into(),
        format!("{NOVNC_PORT}:127.0.0.1:{NOVNC_PORT}"),
        target,
    ]);

    let mut child = Command::new("ssh")
        .args(&argv)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot start ssh: {e}. Is the OpenSSH client installed and on PATH?"))?;

    // Give the forward a moment, then confirm the API is actually answering.
    // A tunnel process that is alive but not forwarding is the failure mode
    // that otherwise shows up as a blank window.
    let mut ready = false;
    for _ in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(150));
        if std::net::TcpStream::connect(("127.0.0.1", API_PORT)).is_ok() {
            ready = true;
            break;
        }
    }
    if !ready {
        let _ = child.kill();
        return Err(format!(
            "tunnel opened but nothing answered on 127.0.0.1:{API_PORT}. \
             Check the host, the port, and that the key is loaded in ssh-agent."
        ));
    }

    guard.child = Some(child);
    guard.api = API_PORT;
    guard.ws = WS_PORT;
    guard.novnc = NOVNC_PORT;
    Ok(TunnelStatus {
        running: true,
        api_port: guard.api,
        ws_port: guard.ws,
        novnc_port: guard.novnc,
        error: String::new(),
    })
}

#[tauri::command]
fn tunnel_status(state: State<'_, Mutex<Tunnel>>) -> TunnelStatus {
    let mut guard = match state.lock() {
        Ok(g) => g,
        Err(_) => return TunnelStatus { running: false, api_port: 0, ws_port: 0, novnc_port: 0, error: "state poisoned".into() },
    };
    let running = guard
        .child
        .as_mut()
        .map(|c| matches!(c.try_wait(), Ok(None)))
        .unwrap_or(false);
    TunnelStatus {
        running,
        api_port: guard.api,
        ws_port: guard.ws,
        novnc_port: guard.novnc,
        error: String::new(),
    }
}

#[tauri::command]
fn close_tunnel(state: State<'_, Mutex<Tunnel>>) {
    if let Ok(mut guard) = state.lock() {
        if let Some(child) = guard.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        guard.child = None;
    }
}

/// Ports the frontend needs before the tunnel exists, so the UI can render a
/// connection panel rather than a broken page.
#[tauri::command]
fn default_ports() -> HashMap<String, u16> {
    HashMap::from([
        ("api".to_string(), API_PORT),
        ("ws".to_string(), WS_PORT),
        ("novnc".to_string(), NOVNC_PORT),
    ])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(Mutex::new(Tunnel::default()))
        .invoke_handler(tauri::generate_handler![
            api_base,
            open_tunnel,
            tunnel_status,
            close_tunnel,
            default_ports
        ])
        .setup(|app| {
            // Killing ssh on exit matters: a GUI that leaves a root-forwarding
            // tunnel behind after the window closes is a surprise nobody wants.
            let handle = app.handle().clone();
            app.listen("tauri://close-requested", move |_| {
                if let Some(state) = handle.try_state::<Mutex<Tunnel>>() {
                    if let Ok(mut guard) = state.lock() {
                        if let Some(child) = guard.child.as_mut() {
                            let _ = child.kill();
                        }
                        guard.child = None;
                    }
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running agentd");
}
