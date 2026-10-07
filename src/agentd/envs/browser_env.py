"""Browser environment: the agent acts on a real page through Playwright.

The observation is a numbered element list in the style of BrowserGym/WebArena
(`[1] button "Add to cart"`), which keeps the action space closed and enumerable —
the advantage term can only rank a set it can see. Actions are element-indexed
(`click [3]`, `type [5] some text`) rather than pixel coordinates, because a
coordinate policy cannot be compared across runs or reused as experience.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..context import compact_observation

ELEMENT_JS = """
() => {
  const sel = 'a[href],button,input,select,textarea,[role=button],[onclick],'
            + '[type=submit],[type=button],[contenteditable="true"],label';
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    const rect = el.getBoundingClientRect();
    const visible = rect.width > 0 && rect.height > 0 && getComputedStyle(el).visibility !== 'hidden';
    if (!visible) continue;
    const text = (el.innerText || el.value || el.getAttribute('aria-label')
                  || el.getAttribute('placeholder') || el.getAttribute('title')
                  || el.getAttribute('alt') || '').trim().replace(/\\s+/g, ' ').slice(0, 70);
    out.push({
      tag: el.tagName.toLowerCase(),
      type: (el.type || '').toLowerCase(),
      text,
      name: el.name || '',
      id: el.id || '',
      role: el.getAttribute('role') || '',
      disabled: !!el.disabled,
    });
    if (out.length >= 60) break;
  }
  return out;
}
"""

BODY_TEXT_JS = "() => document.body ? document.body.innerText.slice(0, 3000) : ''"

CLICK_JS = """
(args) => {
  const [index, selector] = args;
  const els = [...document.querySelectorAll(selector)];
  const el = els[index];
  if (!el) return 'missing';
  el.scrollIntoView({block: 'center'});
  el.click();
  return 'clicked ' + el.tagName.toLowerCase();
}
"""

SELECTOR = ('a[href],button,input,select,textarea,[role=button],[onclick],'
            '[type=submit],[type=button],[contenteditable="true"],label')

_CLICK = re.compile(r"^click\s+\[(\d+)\]", re.I)
_TYPE = re.compile(r'^type\s+\[(\d+)\]\s+(.*)$', re.I)
_SELECT = re.compile(r"^select\s+\[(\d+)\]\s+(.*)$", re.I)


@dataclass
class WebTask:
    """A page plus a success condition evaluated inside it."""

    name: str
    goal: str
    html: str = ""
    url: str = ""
    checker: Callable[[object], bool] | None = None      # receives the Playwright page
    max_steps: int = 10
    scope: str = ""

    def __post_init__(self):
        self.scope = self.scope or f"web:{self.name}"


class BrowserEnv:
    """Implements the Task protocol over a Playwright page."""

    def __init__(self, page, task: WebTask, observation_chars: int = 3200):
        self.page = page
        self.task = task
        self.observation_chars = observation_chars
        self.elements: list[dict] = []
        self.steps = 0
        self.finished = False
        self.reward = 0.0

    # -- Task protocol: name/scope live on the env so the loop can group traces
    @property
    def name(self) -> str:
        return self.task.name

    @property
    def scope(self) -> str:
        return self.task.scope

    # -- Task protocol ----------------------------------------------------
    def reset(self) -> str:
        if self.task.url:
            self.page.goto(self.task.url, wait_until="domcontentloaded")
        else:
            self.page.set_content(self.task.html)
        self.steps = 0
        self.finished = False
        self.reward = 0.0
        return self.observe()

    def goal(self) -> str:
        return self.task.goal

    def observe(self) -> str:
        self.elements = self.page.evaluate(ELEMENT_JS) or []
        try:
            body, _ = compact_observation(self.page.evaluate(BODY_TEXT_JS) or "",
                                          self.observation_chars)
        except Exception:
            body = ""
        lines = [f"# page: {self.page.title() or self.task.name}", f"# url: {self.page.url}"]
        if body:
            lines.append(f"# visible text\n{body}")
        lines.append("# elements")
        for i, element in enumerate(self.elements, start=1):
            kind = element["type"] or element["role"] or element["tag"]
            label = element["text"] or element["name"] or element["id"] or "(unlabelled)"
            flag = " disabled" if element["disabled"] else ""
            lines.append(f"[{i}] {kind} {element['tag']}: {label[:70]}{flag}")
        return "\n".join(lines)

    def candidates(self, state: str) -> list[str]:
        options: list[str] = []
        for i, element in enumerate(self.elements, start=1):
            if element["disabled"]:
                continue
            kind = element["type"] or element["role"] or element["tag"]
            label = (element["text"] or element["name"] or element["id"] or "(unlabelled)")[:60]
            if kind in ("text", "search", "email", "tel", "url", "password", "number", "textarea"):
                options.append(f'type [{i}] <value for {label}>')
            elif kind == "select-one" or element["tag"] == "select":
                options.append(f"select [{i}] <option>")
            elif kind in ("submit", "button") or element["tag"] in ("button", "a"):
                options.append(f"click [{i}] {label}")
        if options:
            options.append("scroll down")
            options.append("wait and re-read the page")
        return options[:40]

    def apply(self, action: str, ctx=None) -> tuple[str, bool, float]:
        self.steps += 1
        outcome = self._dispatch(action)
        success = self._check_success()
        if success:
            self.reward, self.finished = 1.0, True
            return self.observe(), True, 1.0
        if self.steps >= self.task.max_steps:
            self.finished = True
            return self.observe(), True, 0.0
        return self.observe(), False, 0.0

    # -- internals --------------------------------------------------------
    def _dispatch(self, action: str) -> str:
        text = (action or "").strip()
        lowered = text.lower()

        if lowered.startswith("click"):
            match = _CLICK.match(text)
            if not match:
                return "malformed click"
            index = int(match.group(1)) - 1
            label = text[match.end():].strip().lower()
            index = self._resolve(index, label)
            if index is None:
                return "no such element"
            return self.page.evaluate(CLICK_JS, [index, SELECTOR])

        if lowered.startswith("type"):
            match = _TYPE.match(text)
            if not match:
                return "malformed type"
            index, value = int(match.group(1)) - 1, match.group(2).strip()
            if index < 0 or index >= len(self.elements):
                return "no such element"
            value = _placeholder_value(value)
            try:
                self.page.evaluate(
                    "([i, sel, v]) => { const el = [...document.querySelectorAll(sel)][i];"
                    " if (!el) return; el.focus(); el.value = v;"
                    " el.dispatchEvent(new Event('input', {bubbles: true}));"
                    " el.dispatchEvent(new Event('change', {bubbles: true})); }",
                    [index, SELECTOR, value],
                )
                return f"typed into [{index + 1}]"
            except Exception as exc:
                return f"type failed: {exc}"

        if lowered.startswith("select"):
            match = _SELECT.match(text)
            if not match:
                return "malformed select"
            index, value = int(match.group(1)) - 1, match.group(2).strip()
            try:
                self.page.select_option(f"css={SELECTOR} >> nth={index}", _placeholder_value(value))
                return f"selected [{index + 1}]"
            except Exception as exc:
                return f"select failed: {exc}"

        if lowered.startswith("press"):
            key = text.split(None, 1)[1] if " " in text else "Enter"
            try:
                self.page.keyboard.press(key)
                return f"pressed {key}"
            except Exception as exc:
                return f"press failed: {exc}"

        if lowered.startswith("goto"):
            url = text.split(None, 1)[1].strip() if " " in text else ""
            allowed = getattr(self.task, "allowed_origins", ())
            if allowed and not any(url.startswith(origin) for origin in allowed):
                return f"blocked navigation to {url} (not in allowed_origins)"
            try:
                self.page.goto(url, wait_until="domcontentloaded")
                return f"navigated to {url}"
            except Exception as exc:
                return f"goto failed: {exc}"

        if lowered.startswith("scroll"):
            self.page.evaluate("() => window.scrollBy(0, 600)")
            return "scrolled"

        if lowered.startswith("wait"):
            self.page.wait_for_timeout(150)
            return "waited"

        if lowered in ("done", "finish", "submit task"):
            return "declared done"
        return f"unknown action: {text[:40]}"

    def _resolve(self, index: int, label: str) -> int | None:
        """Map an action's index to the element it *meant*.

        Indices are positional, and the page changes under the agent. Because the
        kernel replays action strings from memory, a stale index would silently
        click something else — so when the label no longer matches what sits at
        that index, fall back to the first element whose label matches.
        """
        if not (0 <= index < len(self.elements)):
            return index if 0 <= index < len(self.elements) else self._by_label(label)
        if not label:
            return index
        current = (self.elements[index].get("text") or self.elements[index].get("name")
                   or self.elements[index].get("id") or "").lower()
        if label in current or current in label or not current:
            return index
        return self._by_label(label) or index

    def _by_label(self, label: str) -> int | None:
        if not label:
            return None
        for i, element in enumerate(self.elements):
            text = (element.get("text") or element.get("name") or element.get("id") or "").lower()
            if text and (label in text or text in label):
                return i
        return None

    def _check_success(self) -> bool:
        if self.task.checker is None:
            return False
        try:
            return bool(self.task.checker(self.page))
        except Exception:
            return False


def _placeholder_value(value: str) -> str:
    """Models sometimes echo the `<value for X>` placeholder instead of filling it in.

    Falling back to a plausible string keeps a run moving instead of dead-ending on
    a formatting slip.
    """
    stripped = value.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return "test value"
    return stripped
