"""The conversational agent: OpenAI tool calling over policy search (RAG) and the HR MCP session.

The model chooses the tool; the code runs it. Policy questions go to the Step 6
Answerer, so citations and source precedence stay enforced in code. Leave
actions go through HRSession, which fixes the employee ID and refuses hr_* tools.
"""

import asyncio
import json
from datetime import date, timedelta
from typing import Protocol

from openai import OpenAI

from assistant.hr_session import HRSession, error
from hr_mcp.config import app_today
from rag.answering import Answer
from rag.llm import CHAT_MODEL, make_client

MAX_TOOL_ROUNDS = 6
MAX_HISTORY_MESSAGES = 40
FALLBACK_REPLY = "ბოდიში, მოთხოვნის დამუშავება ვერ მოხერხდა. გთხოვთ, ჩამოაყალიბოთ კითხვა სხვაგვარად."
LEAVE_TYPES = ["ANNUAL", "SICK", "UNPAID", "BEREAVEMENT", "STUDY", "PARENTAL"]
REQUEST_STATUSES = ["pending", "approved", "rejected", "cancelled"]
WEEKDAYS = ["ორშაბათი", "სამშაბათი", "ოთხშაბათი", "ხუთშაბათი", "პარასკევი", "შაბათი", "კვირა"]

SYSTEM_PROMPT = """You are the HR assistant of Northstar Services in a Georgian-language command-line app.
You are talking to employee {employee_id}. Today is {today} ({weekday}). Use this date, never your own clock.
Tomorrow (ხვალ) is {tomorrow}. Next week (მომავალი კვირა) runs from {next_monday} to {next_sunday}.
Calendar until the end of the year. Always read dates from it, never compute weekdays yourself:
{calendar}

Always reply in Georgian, using only Georgian script. Be short and clear.

Choosing a tool:
- Any question about company rules, limits, deadlines, rights or procedures (leave, remote work, travel,
  expenses, training, IT security, conduct): call search_policies. Never answer such questions from your own
  knowledge. Show the tool's answer as it is, keeping every citation in parentheses, the "შენიშვნა" line and
  the source lists; you may add one short lead-in sentence. If the tool found nothing, say so.
- "How many days do I have": call get_leave_balance. Report the available days; mention pending days if any,
  because the balance already subtracts approved and pending requests.
- The employee's own requests: call list_my_leave_requests.
- A question that needs both (e.g. "how many days do I have and how early must I ask?"): call both tools.
- Never add article numbers or citations yourself; only repeat the ones that appear in tool results.

Creating a leave request:
1. You need the leave type and the start and end dates (YYYY-MM-DD); for UNPAID also a brief reason.
   Ask only for what is missing; never guess dates. Plain "შვებულება" means ANNUAL.
   Types: ყოველწლიური = ANNUAL, ავადმყოფობის = SICK, უხელფასო = UNPAID, გლოვის / ახლობლის გარდაცვალება =
   BEREAVEMENT, სასწავლო = STUDY, დეკრეტული / მშობლის = PARENTAL.
2. As soon as you have them, call propose_leave_request. Do NOT ask for confirmation before proposing.
   Use exactly the type the employee needs; never switch to another type (e.g. never UNPAID instead of
   BEREAVEMENT). For BEREAVEMENT, STUDY and PARENTAL still call propose_leave_request: the tool answers where
   they must be submitted, because the assistant can create only ANNUAL, SICK and UNPAID.
3. A successful proposal is NOT a request yet: say it is prepared but not submitted, show the type, dates,
   number of working days and the balance after, and ask the employee to confirm. Never say "created" or
   "submitted" before confirm_leave_request succeeds. Call it only after the employee's next message clearly confirms.
   If they change something, propose again. After confirming, give the request ID and say it is pending
   approval, not approved.
If a tool returns errors, explain each one in Georgian with its article (e.g. "მუხლი 4.4") and where to turn,
using redirect_to: MANAGER = უშუალო ხელმძღვანელი, HR = HR დეპარტამენტი, HR_PORTAL = HR პორტალი,
HR_PORTAL_OR_HR = HR პორტალი ან HR დეპარტამენტი.

Limits:
- Confirming your own proposal ("კი", "ვადასტურებ") is allowed. Approving, rejecting, cancelling or changing
  a submitted request ("დაამტკიცე", "გააუქმე") is not: refuse, cite Article 12.3 and refer the employee to
  the HR portal or their manager. You also cannot see other employees' data (Article 12.3).
- Sick leave: never ask about or record a diagnosis or other health details; leave the comment empty.
  Sick leave records actual illness. If a SICK request starts later than {today}, ask whether it is for a
  planned medical procedure, and pass planned_procedure=true only if the employee says yes."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": "Search the company policy documents and get an answer with citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The policy question in Georgian, self-contained (resolve 'it', 'that' "
                        "from the conversation).",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leave_balance",
            "description": "The employee's remaining leave days (entitled + carried over - approved - pending).",
            "parameters": {
                "type": "object",
                "properties": {
                    "leave_type": {"type": "string", "enum": LEAVE_TYPES, "description": "Omit for all types."},
                    "year": {"type": "integer", "description": "Omit for the current year."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_my_leave_requests",
            "description": "The employee's own leave requests, optionally filtered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": REQUEST_STATUSES},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_leave_types",
            "description": "All leave types: day unit, yearly limit, whether the assistant can create them, "
            "submission channel and policy article.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_leave_request",
            "description": "Step 1: validate a leave request against the policy and prepare it. Creates nothing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "leave_type": {
                        "type": "string",
                        "enum": LEAVE_TYPES,
                        "description": "Plain 'შვებულება' without another type means ANNUAL.",
                    },
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "comment": {
                        "type": "string",
                        "description": "Brief reason; required for UNPAID. Never health details.",
                    },
                    "planned_procedure": {
                        "type": "boolean",
                        "description": "Only for SICK leave starting after today: true once the employee said "
                        "it is for a planned medical procedure.",
                    },
                },
                "required": ["leave_type", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_leave_request",
            "description": "Step 2: create the latest proposal as a pending request. Only after the employee "
            "explicitly confirmed it in their latest message.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class PolicyAnswerer(Protocol):
    def answer(self, question: str) -> Answer: ...


def needs_procedure_question(arguments: dict) -> bool:
    """Sick leave in the future is unusual: the employee must first say it is for a planned procedure."""
    if str(arguments.get("leave_type", "")).upper() != "SICK" or arguments.get("planned_procedure") is True:
        return False
    try:
        return date.fromisoformat(str(arguments.get("start_date", ""))) > app_today()
    except ValueError:
        return False


def day_label(day: date) -> str:
    return f"{day.isoformat()} {WEEKDAYS[day.weekday()]}"


def calendar_lines(today: date) -> str:
    """One line per week from this week's Monday to December 31, e.g. '2026-10-19 ორშაბათი | ...'."""
    lines = []
    day = today - timedelta(days=today.weekday())
    while day.year == today.year:
        week = [day + timedelta(days=i) for i in range(7)]
        lines.append(" | ".join(day_label(d) for d in week if d.year == today.year))
        day += timedelta(days=7)
    return "\n".join(lines)


class Agent:
    def __init__(
        self, session: HRSession, answerer: PolicyAnswerer, model: str = CHAT_MODEL, client: OpenAI | None = None
    ):
        self.session = session
        self.answerer = answerer
        self.model = model
        self.client = client or make_client()
        today = app_today()
        next_monday = today + timedelta(days=7 - today.weekday())
        self.system_prompt = SYSTEM_PROMPT.format(
            employee_id=session.employee_id,
            today=today.isoformat(),
            weekday=WEEKDAYS[today.weekday()],
            tomorrow=day_label(today + timedelta(days=1)),
            next_monday=day_label(next_monday),
            next_sunday=day_label(next_monday + timedelta(days=6)),
            calendar=calendar_lines(today),
        )
        self.history: list[dict] = []

    async def reply(self, text: str) -> str:
        self.session.new_turn()
        self.history.append({"role": "user", "content": text})
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.model, temperature=0, tools=TOOLS, messages=self._messages()
            )
            message = response.choices[0].message
            if not message.tool_calls:
                answer = (message.content or "").strip() or FALLBACK_REPLY
                self.history.append({"role": "assistant", "content": answer})
                return answer

            self.history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in message.tool_calls
                ],
            })
            for call in message.tool_calls:
                result = await self.run_tool(call.function.name, call.function.arguments)
                self.history.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}
                )
        self.history.append({"role": "assistant", "content": FALLBACK_REPLY})
        return FALLBACK_REPLY

    async def run_tool(self, name: str, raw_arguments: str) -> dict:
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            return error("BAD_ARGUMENTS", "Tool arguments must be valid JSON.")
        if not isinstance(arguments, dict):
            return error("BAD_ARGUMENTS", "Tool arguments must be a JSON object.")
        try:
            match name:
                case "search_policies":
                    answer = await asyncio.to_thread(self.answerer.answer, str(arguments.get("question", "")))
                    return {"found": answer.found, "answer": answer.format()}
                case "get_leave_balance":
                    return await self.session.balance(arguments.get("leave_type"), arguments.get("year"))
                case "list_my_leave_requests":
                    return await self.session.my_requests(
                        arguments.get("status"), arguments.get("date_from"), arguments.get("date_to")
                    )
                case "list_leave_types":
                    return await self.session.leave_types()
                case "propose_leave_request":
                    if needs_procedure_question(arguments):
                        return error(
                            "ASK_EMPLOYEE_FIRST",
                            "This SICK leave starts after today. Ask the employee whether it is for a planned "
                            "medical procedure (no other health details); if yes, call again with "
                            "planned_procedure=true.",
                        )
                    return await self.session.propose(
                        arguments.get("leave_type", ""),
                        arguments.get("start_date", ""),
                        arguments.get("end_date", ""),
                        arguments.get("comment"),
                    )
                case "confirm_leave_request":
                    return await self.session.confirm()
                case _:
                    return error("UNKNOWN_TOOL", f"There is no tool named {name}.")
        except PermissionError as exc:
            return error("FORBIDDEN", str(exc))

    def _messages(self) -> list[dict]:
        """System prompt plus recent history, cut at a user message so tool results never lose their call."""
        history = self.history[-MAX_HISTORY_MESSAGES:]
        while history and history[0]["role"] != "user":
            history = history[1:]
        return [{"role": "system", "content": self.system_prompt}, *history]
