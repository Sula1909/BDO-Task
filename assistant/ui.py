"""Terminal presentation with rich: panels, the confirmation card and Georgian error messages.

The proposal and request cards are built from the MCP tool results, not from the
model's text, so the dates and day counts the employee confirms are exactly what
the server validated.
"""

from datetime import date

import openai
from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DAY_UNITS = {"working": "სამუშაო დღე", "calendar": "კალენდარული დღე"}
STATUSES = {
    "pending": "ელოდება დამტკიცებას",
    "approved": "დამტკიცებული",
    "rejected": "უარყოფილი",
    "cancelled": "გაუქმებული",
}
COMMANDS = {
    "/help": "დახმარება",
    "/new": "ახალი საუბრის დაწყება",
    "/exit": "გასვლა",
}


def welcome_panel(employee_id: str, today: date) -> Panel:
    body = Group(
        Text.assemble(("თანამშრომელი: ", "dim"), (employee_id, "bold"), ("   თარიღი: ", "dim"), today.isoformat()),
        Text(""),
        help_text(),
    )
    return Panel(body, title="[bold]Northstar Services · HR ასისტენტი[/]", border_style="cyan", box=box.ROUNDED)


def help_text() -> Group:
    examples = Table.grid(padding=(0, 2))
    examples.add_row("[cyan]პოლიტიკა[/]", "„რამდენი დღე შემიძლია ვიმუშაო სახლიდან?“")
    examples.add_row("[cyan]ბალანსი[/]", "„რამდენი შვებულების დღე დამრჩა?“")
    examples.add_row("[cyan]მოთხოვნა[/]", "„შვებულება მინდა 2-დან 4 ნოემბრამდე“")
    commands = Table.grid(padding=(0, 2))
    for command, description in COMMANDS.items():
        commands.add_row(f"[bold]{command}[/]", description)
    return Group(Text("რა შემიძლია:", style="bold"), examples, Text(""), Text("ბრძანებები:", style="bold"), commands)


def help_panel() -> Panel:
    return Panel(help_text(), title="დახმარება", border_style="cyan", box=box.ROUNDED)


def reply_panel(text: str) -> Panel:
    return Panel(Markdown(text), title="ასისტენტი", title_align="left", border_style="blue", box=box.ROUNDED)


def _details_table(rows: list[tuple[str, str]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    for label, value in rows:
        table.add_row(label, value)
    return table


def proposal_panel(result: dict) -> Panel:
    """The card shown while a proposal waits for the employee's 'yes'."""
    p = result["proposal"]
    unit = DAY_UNITS.get(p.get("day_unit") or "", "დღე")
    rows = [
        ("სახე", f"{p.get('leave_type_name') or p['leave_type']} ({p['leave_type']})"),
        ("პერიოდი", f"{p['start_date']} — {p['end_date']}"),
        ("ხანგრძლივობა", f"{p['days']} {unit}"),
        ("ბალანსი", f"{result['available_before']} → {result['available_after']}"),
    ]
    if p.get("comment"):
        rows.append(("კომენტარი", p["comment"]))
    footer = Text.assemble(
        "მოთხოვნა ჯერ არ არის გაგზავნილი. დასადასტურებლად დაწერეთ ", ("კი", "bold green"),
        ", გასაუქმებლად ", ("არა", "bold red"), ".",
    )
    return Panel(
        Group(_details_table(rows), Text(""), footer),
        title="[bold yellow]⚠ დასადასტურებელი მოთხოვნა[/]",
        border_style="yellow",
        box=box.HEAVY,
        padding=(1, 2),
    )


def request_panel(result: dict) -> Panel:
    """The card shown after confirm_leave_request created (or had already created) the request."""
    r = result["request"]
    rows = [
        ("მოთხოვნის ნომერი", f"#{r['request_id']}"),
        ("სახე", r["leave_type"]),
        ("პერიოდი", f"{r['start_date']} — {r['end_date']}"),
        ("დღეები", str(r["days"])),
        ("სტატუსი", STATUSES.get(r["status"], r["status"])),
    ]
    title = "მოთხოვნა უკვე გაგზავნილია" if result.get("already_confirmed") else "მოთხოვნა გაგზავნილია"
    return Panel(_details_table(rows), title=f"[bold green]✔ {title}[/]", border_style="green", box=box.HEAVY,
                 padding=(1, 2))


def sources_panel(results: list[dict]) -> Panel | None:
    """Sources, outdated sources and conflict notes of this turn's policy searches; None if there is nothing."""

    def unique(key: str) -> list[dict]:
        seen: dict[str, dict] = {}
        for result in results:
            for item in result.get(key) or []:
                seen.setdefault(item["citation"], item)
        return list(seen.values())

    sources, outdated = unique("sources"), unique("outdated")
    notes = list(dict.fromkeys(r["conflict_note"] for r in results if r.get("conflict_note")))
    if not (sources or outdated or notes):
        return None

    parts: list = []
    for note in notes:
        parts.append(Text.assemble(("შენიშვნა: ", "bold yellow"), note))
    if sources:
        parts.append(Text("წყაროები:", style="bold"))
        parts += [Text.assemble("  • ", (s["citation"], "cyan"), f" — {s['title']}") for s in sources]
    if outdated:
        parts.append(Text("მოძველებული ინფორმაცია (არ გამოიყენება):", style="bold"))
        parts += [Text.assemble("  • ", (o["citation"], "dim strike"), f" — {o['title']}") for o in outdated]
    return Panel(Group(*parts), title="წყაროები", title_align="left", border_style="dim", box=box.ROUNDED)


def error_panel(message: str, hint: str = "") -> Panel:
    body = Text(message)
    if hint:
        body.append(f"\n\n{hint}", style="dim")
    return Panel(body, title="[bold red]შეცდომა[/]", border_style="red", box=box.ROUNDED)


def describe_error(exc: BaseException) -> tuple[str, str]:
    """(message, hint) in Georgian for an error raised while answering. Never shows the API key."""
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI-მ API გასაღები არ მიიღო.", "შეამოწმეთ OPENAI_API_KEY პროექტის .env ფაილში და გაუშვით თავიდან."
    if isinstance(exc, openai.PermissionDeniedError):
        return "OpenAI-მ მოთხოვნა უარყო (წვდომა აკრძალულია).", "შეამოწმეთ, აქვს თუ არა გასაღებს მოდელზე წვდომა."
    if isinstance(exc, openai.RateLimitError):
        return (
            "OpenAI-ის მოთხოვნების ლიმიტი ამოიწურა ან ანგარიშზე თანხა არ არის.",
            "მოიცადეთ ცოტა ხანი და სცადეთ ხელახლა.",
        )
    if isinstance(exc, openai.APITimeoutError):
        return "OpenAI-მ დროულად არ უპასუხა.", "სცადეთ ხელახლა."
    if isinstance(exc, openai.APIConnectionError):
        return "OpenAI-სთან კავშირი ვერ დამყარდა.", "შეამოწმეთ ინტერნეტ კავშირი და სცადეთ ხელახლა."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI-ის სერვისმა შეცდომა დააბრუნა (კოდი {exc.status_code}).", "სცადეთ ცოტა ხანში."
    return "მოულოდნელი შეცდომა.", f"{type(exc).__name__}. თუ განმეორდა, გადატვირთეთ პროგრამა."
