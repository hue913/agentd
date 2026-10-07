"""Small web tasks with deterministic success checks — no network, no dataset download.

They are deliberately MiniWoB-shaped: a closed set of numbered elements, one goal,
a checker that reads the DOM. That is what makes the learning curve measurable:
reward comes from the page, not from a model grading itself.
"""

from __future__ import annotations

from .browser_env import WebTask

CONTACT_FORM = """<!doctype html><html><head><title>Contact support</title></head><body>
<h2>Contact support</h2>
<input id="name" placeholder="Your name">
<input id="email" placeholder="Email address">
<textarea id="msg" placeholder="Describe the problem"></textarea>
<button id="send" onclick="go()">Send message</button>
<div id="done"></div>
<script>function go(){
  if(document.getElementById('name').value && document.getElementById('email').value.includes('@')
     && document.getElementById('msg').value){
    document.getElementById('done').textContent='Thanks, ticket opened';
  } else { document.getElementById('done').textContent='fill everything first'; }
}</script></body></html>"""

MODAL_GATE = """<!doctype html><html><head><title>Archive</title></head><body>
<h2>Reports</h2>
<div id="overlay" style="position:fixed;inset:0;background:#eee">
  <p>We use cookies</p><button id="accept" onclick="close1()">Accept cookies</button>
</div>
<button id="download" onclick="dl()">Download report</button>
<div id="log"></div>
<script>function close1(){document.getElementById('overlay').remove();
  document.getElementById('log').textContent='overlay dismissed';}
function dl(){ if(document.getElementById('overlay')) { document.getElementById('log').textContent='blocked'; return;}
  document.getElementById('log').textContent='report downloaded';}
function dl2(){dl();}</script></body></html>"""

CHEAPEST_BUY = """<!doctype html><html><head><title>Store</title></head><body>
<h2>Keyboards</h2>
<div class="row"><span>Model K200 — $89</span><button onclick="buy('k200',89)">Buy K200</button></div>
<div class="row"><span>Model Mini — $45</span><button onclick="buy('mini',45)">Buy Mini</button></div>
<div class="row"><span>Model Pro X — $140</span><button onclick="buy('pro',140)">Buy Pro X</button></div>
<input id="qty" placeholder="Quantity">
<button id="checkout" onclick="co()">Proceed to checkout</button>
<div id="receipt"></div>
<script>let cart=null;
function buy(id,price){cart=id; document.getElementById('receipt').textContent='cart='+id;}
function co(){ if(!cart){document.getElementById('receipt').textContent='cart empty';return;}
  document.getElementById('receipt').textContent='ordered='+cart+':'+document.getElementById('qty').value;}</script>
</body></html>"""

TOGGLE_SUM = """<!doctype html><html><head><title>Pick items</title></head><body>
<h2>Choose items totalling exactly 12</h2>
<label><input type="checkbox" value="5" onchange="sum()"> item worth 5</label>
<label><input type="checkbox" value="7" onchange="sum()"> item worth 7</label>
<label><input type="checkbox" value="9" onchange="sum()"> item worth 9</label>
<label><input type="checkbox" value="12" onchange="sum()"> item worth 12</label>
<div id="total">total=0</div>
<button id="submit" onclick="done()">Submit selection</button>
<script>function sum(){let t=0;document.querySelectorAll('input:checked').forEach(e=>t+=+e.value);
  document.getElementById('total').textContent='total='+t;}
function done(){let t=0;document.querySelectorAll('input:checked').forEach(e=>t+=+e.value);
  document.getElementById('total').textContent = (t===12) ? 'SUBMITTED total=12' : 'SUBMITTED wrong total='+t;}</script>
</body></html>"""

LOGIN_GATE = """<!doctype html><html><head><title>Console</title></head><body>
<h2>Sign in to export</h2>
<input id="user" placeholder="username">
<input id="pass" type="password" placeholder="password">
<button id="signin" onclick="login()">Sign in</button>
<button id="export" style="display:none" onclick="exp()">Export CSV</button>
<div id="out"></div>
<script>function login(){
  if(document.getElementById('user').value && document.getElementById('pass').value){
    document.getElementById('export').style.display='inline';
    document.getElementById('out').textContent='signed in';}
  else {document.getElementById('out').textContent='need both fields';}}
function exp(){document.getElementById('out').textContent='exported.csv downloaded';}</script>
</body></html>"""

WIZARD = """<!doctype html><html><head><title>Setup wizard</title></head><body>
<div id="step1"><h2>Step 1: region</h2>
  <button onclick="next('eu')">Europe</button><button onclick="next('us')">US</button>
  <button onclick="bad()">Skip everything</button></div>
<div id="step2" style="display:none"><h2>Step 2: plan</h2>
  <button onclick="next2('pro')">Pro</button><button onclick="next2('free')">Free</button></div>
<div id="step3" style="display:none"><h2>Step 3: confirm</h2>
  <button onclick="fin()">Create workspace</button></div>
<div id="state"></div>
<script>let region=null,plan=null;
function next(r){region=r;document.getElementById('step1').style.display='none';
  document.getElementById('step2').style.display='block';document.getElementById('state').textContent='region='+r;}
function bad(){document.getElementById('state').textContent='skipped everything (nothing created)';}
function next2(p){plan=p;document.getElementById('step2').style.display='none';
  document.getElementById('step3').style.display='block';document.getElementById('state').textContent='region='+region+' plan='+p;}
function fin(){document.getElementById('state').textContent='created region='+region+' plan='+plan;}</script>
</body></html>"""


def _text_of(page, selector: str) -> str:
    try:
        return page.inner_text(selector)
    except Exception:
        return ""


def contact_task() -> WebTask:
    return WebTask(
        name="contact-form",
        goal="Open a support ticket: fill the name, a valid email, describe the problem, then send.",
        html=CONTACT_FORM,
        max_steps=10,
        checker=lambda page: "Thanks, ticket opened" in _text_of(page, "#done"),
    )


def modal_task() -> WebTask:
    return WebTask(
        name="dismiss-then-download",
        goal="Download the report. A cookie banner covers the page and must be dismissed first.",
        html=MODAL_GATE,
        max_steps=8,
        checker=lambda page: "report downloaded" in _text_of(page, "#log"),
    )


def cheapest_task() -> WebTask:
    return WebTask(
        name="buy-cheapest",
        goal="Buy the cheapest keyboard, set a quantity, and proceed to checkout.",
        html=CHEAPEST_BUY,
        max_steps=10,
        checker=lambda page: "ordered=mini:1" in _text_of(page, "#receipt")
        or _text_of(page, "#receipt").startswith("ordered=mini:"),
    )


def toggle_task() -> WebTask:
    return WebTask(
        name="sum-to-twelve",
        goal="Select checkboxes totalling exactly 12 and submit.",
        html=TOGGLE_SUM,
        max_steps=10,
        checker=lambda page: "SUBMITTED total=12" in _text_of(page, "#total"),
    )


def login_task() -> WebTask:
    return WebTask(
        name="login-export",
        goal="Sign in with any username and password, then export the CSV.",
        html=LOGIN_GATE,
        max_steps=8,
        checker=lambda page: "exported.csv downloaded" in _text_of(page, "#out"),
    )


def wizard_task() -> WebTask:
    return WebTask(
        name="setup-wizard",
        goal="Create a workspace in Europe with the Pro plan via the wizard. Do not skip.",
        html=WIZARD,
        max_steps=10,
        checker=lambda page: "created region=eu plan=pro" in _text_of(page, "#state"),
    )


WEB_TASKS = {
    "contact-form": contact_task,
    "dismiss-then-download": modal_task,
    "buy-cheapest": cheapest_task,
    "sum-to-twelve": toggle_task,
    "login-export": login_task,
    "setup-wizard": wizard_task,
}


def web_suite() -> list:
    return [factory() for factory in WEB_TASKS.values()]
