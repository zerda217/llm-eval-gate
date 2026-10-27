"""Generate the synthetic Telegram conversations used to evaluate appointment extraction.

The expected answer for every case is *computed* from the same parameters that
produced the conversation (dates, times, who agreed, what was cancelled), so the
ground truth cannot drift from the text. Run from the repo root:

    python examples/schedule/generate.py

Outputs (all deterministic, seed below):
    cases.jsonl                         eval dataset
    fixtures/reference_outputs.jsonl    expected answers as model outputs -> gate must pass
    fixtures/broken_outputs.jsonl       typical LLM mistakes injected     -> gate must block
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

SEED = 217
HERE = Path(__file__).resolve().parent
ME = "나"
WD = "월화수목금토일"

FRIENDS = ["민지", "준호", "서연", "지훈", "하은", "도윤", "수빈", "현우", "예린", "태민"]
PLACES = ["강남역", "홍대입구역", "성수동 카페", "합정역 2번 출구", "판교역", "잠실 롯데월드몰", "을지로3가역", "신촌 스터디카페"]
# (invitation phrase, title noun, allowed start hours, places that fit the activity)
ACTIVITIES = [
    ("저녁 먹자", "저녁", [18, 19, 20], ["강남역", "홍대입구역", "을지로3가역", "합정역 2번 출구"]),
    ("점심 먹자", "점심", [11, 12, 13], ["강남역", "판교역", "을지로3가역", "성수동 카페"]),
    ("커피 한잔 하자", "커피", [10, 14, 15, 16], ["성수동 카페", "강남역", "판교역"]),
    ("영화 보자", "영화", [14, 16, 19, 20], ["잠실 롯데월드몰", "홍대입구역", "강남역"]),
    ("스터디 하자", "스터디", [10, 13, 15, 19], ["신촌 스터디카페", "판교역", "성수동 카페"]),
]
ACCEPT = ["좋아!", "콜 👍", "ㅇㅋ 그때 봐", "좋아 그때 보자", "넵 좋아요", "오 좋아 거기서 봐"]


# ------------------------------------------------------------------ text helpers


def _final_consonant(word: str) -> int:
    """Index of the final consonant (받침) of the last syllable; 0 means none."""
    code = ord(word[-1]) - 0xAC00
    return code % 28 if 0 <= code <= 11171 else 0


def josa_wa(name: str) -> str:
    """'민지와' vs '태민과'."""
    return name + ("과" if _final_consonant(name) else "와")


def josa_ro(word: str) -> str:
    """'7시로', '판교역으로', '롯데월드몰로' (ㄹ받침 takes 로)."""
    jong = _final_consonant(word)
    return word + ("으로" if jong and jong != 8 else "로")


def time_phrase(t: time) -> str:
    h, m = t.hour, t.minute
    if h < 12:
        s = f"오전 {h}시"
    elif h == 12:
        s = "낮 12시"
    elif h < 17:
        s = f"오후 {h - 12}시"
    else:
        s = f"저녁 {h - 12}시"
    return s + (" 반" if m == 30 else "")


def monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def stamp(dt: datetime) -> str:
    return f"[{dt:%Y-%m-%d} {WD[dt.weekday()]} {dt:%H:%M}]"


# ------------------------------------------------------------------ case model


@dataclass
class Convo:
    friend: str
    start: datetime
    lines: list[tuple[datetime, str, str]] = field(default_factory=list)
    _now: datetime | None = None

    def say(self, who: str, text: str, gap_min: int = 2) -> datetime:
        self._now = self.start if self._now is None else self._now + timedelta(minutes=gap_min)
        self.lines.append((self._now, who, text))
        return self._now

    def jump_to(self, dt: datetime) -> None:
        self._now = dt - timedelta(minutes=2)

    def render(self) -> str:
        return "\n".join(f"{stamp(t)} {who}: {text}" for t, who, text in self.lines)


@dataclass
class Plan:
    """One proposed appointment and how the conversation phrases its date."""

    target: date
    start: time | None
    place: str | None
    activity: tuple[str, str, list[int], list[str]]
    date_text: str

    def proposal(self) -> str:
        where = f" {self.place}에서" if self.place else ""
        if self.start:
            return f"{self.date_text} {time_phrase(self.start)}{where} {self.activity[0]}. 어때?"
        when = self.date_text if self.date_text in ("내일", "모레") else self.date_text + "에"
        return f"{when}{where} {self.activity[0]}! 시간은 그날 다시 정하자"

    def event(self, friend: str, *, target: date | None = None, start: time | None | str = "keep") -> dict:
        s = self.start if start == "keep" else start
        return {
            "title": f"{josa_wa(friend)} {self.activity[1]}",
            "date": (target or self.target).isoformat(),
            "start_time": s.strftime("%H:%M") if s else None,
            "location": self.place,
        }


class Gen:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.n = 0
        self.cases: list[dict] = []
        self.broken: dict[str, object] = {}  # case_id -> injected wrong output

    # --- random pieces
    def ref_time(self, min_wd: int = 0, max_wd: int = 6) -> datetime:
        base = date(2026, 10, 26)  # Monday
        while True:
            d = base + timedelta(days=self.rng.randint(0, 20))
            if min_wd <= d.weekday() <= max_wd:
                return datetime.combine(d, time(self.rng.randint(9, 21), self.rng.choice([3, 12, 27, 41, 55])))

    def plan(self, target: date, date_text: str, *, with_time: bool = True, with_place: bool = True) -> Plan:
        act = self.rng.choice(ACTIVITIES)
        start = time(self.rng.choice(act[2]), self.rng.choice([0, 0, 30])) if with_time else None
        return Plan(target, start, self.rng.choice(act[3]) if with_place else None, act, date_text)

    def friend(self) -> str:
        return self.rng.choice(FRIENDS)

    def accept(self) -> str:
        return self.rng.choice(ACCEPT)

    # --- output
    def add(self, convo: Convo, events: list[dict], tags: list[str], broken: object = None) -> str:
        self.n += 1
        cid = f"s{self.n:03d}"
        self.cases.append(
            {"id": cid, "input": {"conversation": convo.render()}, "expected": {"events": events}, "tags": tags}
        )
        if broken is not None:
            self.broken[cid] = broken
        return cid


def shifted(event: dict, days: int = 0, start: str | None = "keep") -> dict:
    e = dict(event)
    e["date"] = (date.fromisoformat(e["date"]) + timedelta(days=days)).isoformat()
    if start != "keep":
        e["start_time"] = start
    return e


# ------------------------------------------------------------------ categories


def absolute(g: Gen, count: int) -> None:
    for i in range(count):
        ref = g.ref_time()
        target = ref.date() + timedelta(days=g.rng.randint(3, 18))
        p = g.plan(target, f"{target.month}월 {target.day}일", with_time=i != 0)
        f = g.friend()
        c = Convo(f, ref)
        c.say(f, "요즘 잘 지내? 얼굴 한번 보자")
        c.say(ME, p.proposal())
        c.say(f, g.accept())
        ev = p.event(f)
        broken = None
        if i == 1 and p.start and p.start.hour >= 13:  # AM/PM confusion
            broken = {"events": [dict(ev, start_time=f"{p.start.hour - 12:02d}:{p.start.minute:02d}")]}
        g.add(c, [ev], ["absolute_date"] + ([] if p.start else ["no_time"]), broken)


def relative_day(g: Gen, count: int) -> None:
    for i in range(count):
        offset, word = (1, "내일") if i % 2 == 0 else (2, "모레")
        cross_midnight = i == count - 1
        ref = g.ref_time()
        if cross_midnight:
            ref = ref.replace(hour=23, minute=51)
        p = g.plan(ref.date() + timedelta(days=offset), word, with_time=i != 2)
        f = g.friend()
        c = Convo(f, ref)
        c.say(f, p.proposal())
        if cross_midnight:
            c.say(ME, g.accept(), gap_min=15)  # reply lands after midnight; "내일" still means the proposer's tomorrow
        else:
            c.say(ME, g.accept())
        ev = p.event(f)
        broken = None
        if cross_midnight:
            broken = {"events": [shifted(ev, days=1)]}  # computed from the reply date instead
        tags = ["relative_day"] + (["cross_midnight"] if cross_midnight else []) + ([] if p.start else ["no_time"])
        g.add(c, [ev], tags, broken)


def weekday(g: Gen, count: int) -> None:
    for i in range(count):
        next_week = i % 2 == 1
        if next_week:
            ref = g.ref_time()
            wd = g.rng.randint(0, 6)
            target = monday(ref.date()) + timedelta(days=7 + wd)
            text = f"다음 주 {WD[wd]}요일"
        else:
            ref = g.ref_time(max_wd=4)  # leave room later in the same week
            wd = g.rng.randint(ref.weekday() + 1, 6)
            target = monday(ref.date()) + timedelta(days=wd)
            text = f"이번 주 {WD[wd]}요일"
        p = g.plan(target, text, with_time=i != 3)
        f = g.friend()
        c = Convo(f, ref)
        c.say(ME, p.proposal())
        c.say(f, g.accept())
        ev = p.event(f)
        broken = {"events": [shifted(ev, days=-7)]} if next_week and i in (1, 5) else None  # read as "this week"
        g.add(c, [ev], ["weekday"] + ([] if p.start else ["no_time"]), broken)


def changed(g: Gen) -> None:
    # 1) time moved and accepted
    ref = g.ref_time()
    target = ref.date() + timedelta(days=4)
    p = g.plan(target, f"{target.month}월 {target.day}일")
    f = g.friend()
    new = time(min(p.start.hour + 1, 21), 0)
    c = Convo(f, ref)
    c.say(f, p.proposal())
    c.say(ME, g.accept())
    c.jump_to(ref + timedelta(hours=26))
    c.say(f, f"아 미안 그날 {josa_ro(time_phrase(new))} 바꿔도 될까? 일이 좀 늦게 끝날 것 같아")
    c.say(ME, "응 괜찮아 그때 봐")
    g.add(c, [p.event(f, start=new)], ["changed"], {"events": [p.event(f)]})

    # 2) postponed one day and accepted
    ref = g.ref_time()
    p = g.plan(ref.date() + timedelta(days=1), "내일")
    f = g.friend()
    c = Convo(f, ref)
    c.say(ME, p.proposal())
    c.say(f, g.accept())
    c.say(f, "아 잠깐, 내일 회식 잡혔다 ㅠ 하루만 미뤄도 돼? 시간이랑 장소는 그대로", gap_min=40)
    c.say(ME, "그래 그럼 모레 같은 시간에 보자")
    g.add(c, [p.event(f, target=p.target + timedelta(days=1))], ["changed"], {"events": [p.event(f)]})

    # 3) counter-proposal rejected -> original stands
    ref = g.ref_time(max_wd=4)
    wd = g.rng.randint(ref.weekday() + 1, 6)
    target = monday(ref.date()) + timedelta(days=wd)
    p = g.plan(target, f"이번 주 {WD[wd]}요일")
    f = g.friend()
    later = time(min(p.start.hour + 1, 21), 0)
    c = Convo(f, ref)
    c.say(f, p.proposal())
    c.say(ME, f"{time_phrase(p.start)} 말고 {time_phrase(later)}는 어때?")
    c.say(f, f"{time_phrase(later)}는 좀 어려워 ㅠ 그냥 원래대로 하자")
    c.say(ME, "ㅇㅋ 그럼 원래 시간에 봐")
    g.add(c, [p.event(f)], ["changed"], {"events": [p.event(f, start=later)]})

    # 4) place changed (date/time unchanged)
    ref = g.ref_time()
    target = ref.date() + timedelta(days=2)
    p = g.plan(target, "모레")
    f = g.friend()
    other = g.rng.choice([x for x in p.activity[3] if x != p.place])
    c = Convo(f, ref)
    c.say(ME, p.proposal())
    c.say(f, g.accept())
    c.say(f, f"아 근데 장소만 {josa_ro(other)} 바꾸자! 거기가 더 가까워", gap_min=30)
    c.say(ME, "좋아")
    ev = p.event(f)
    ev["location"] = other
    g.add(c, [ev], ["changed"])

    # 5) both time and day change across messages
    ref = g.ref_time(max_wd=4)
    target = monday(ref.date()) + timedelta(days=7 + 2)  # next Wednesday
    p = g.plan(target, "다음 주 수요일")
    f = g.friend()
    c = Convo(f, ref)
    c.say(f, p.proposal())
    c.say(ME, g.accept())
    c.say(f, "헉 그 수요일에 출장 생겼어.. 하루 뒤 오후 3시는 어때?", gap_min=90)
    c.say(ME, "그 다음 날 3시 좋아")
    g.add(c, [p.event(f, target=target + timedelta(days=1), start=time(15, 0))], ["changed"], {"events": [p.event(f)]})


def cancelled(g: Gen, count: int) -> None:
    lines = [
        ("미안 ㅠㅠ 그날 갑자기 일이 생겨서 못 갈 것 같아", "아 그래 아쉽다 다음에 보자"),
        ("나 감기 걸려서 이번엔 취소해야 할 것 같아 ㅠ", "헐 몸조리 잘해! 나중에 보자"),
        ("이번 약속은 취소하자.. 일정이 너무 꼬였어", "ㅇㅋ 괜찮아"),
        ("미안한데 그거 없던 걸로 해도 될까?", "응 알겠어"),
    ]
    for i in range(count):
        ref = g.ref_time()
        target = ref.date() + timedelta(days=g.rng.randint(2, 9))
        p = g.plan(target, f"{target.month}월 {target.day}일")
        f = g.friend()
        c = Convo(f, ref)
        c.say(ME, p.proposal())
        c.say(f, g.accept())
        cancel, reply = lines[i % len(lines)]
        c.jump_to(ref + timedelta(hours=g.rng.randint(5, 30)))
        c.say(f, cancel)
        c.say(ME, reply)
        g.add(c, [], ["cancelled"], {"events": [p.event(f)]} if i < 2 else None)


def tentative(g: Gen) -> None:
    # unanswered proposal
    ref = g.ref_time()
    p = g.plan(ref.date() + timedelta(days=3), f"{(ref.date() + timedelta(days=3)).day}일")
    p.date_text = f"{p.target.month}월 {p.target.day}일"
    f = g.friend()
    c = Convo(f, ref)
    c.say(f, "오랜만이야!")
    c.say(f, p.proposal())
    g.add(c, [], ["tentative"], {"events": [p.event(f)]})

    # vague "let's eat sometime"
    ref = g.ref_time()
    f = g.friend()
    c = Convo(f, ref)
    c.say(f, "다음에 밥 한번 먹자~")
    c.say(ME, "그래 언제 한번 보자!")
    g.add(c, [], ["tentative"])

    # pending confirmation
    ref = g.ref_time()
    p = g.plan(ref.date() + timedelta(days=1), "내일")
    f = g.friend()
    c = Convo(f, ref)
    c.say(ME, p.proposal())
    c.say(f, "내일 일정 확인해보고 저녁에 알려줄게!")
    g.add(c, [], ["tentative"], {"events": [p.event(f)]})

    # declined
    ref = g.ref_time(max_wd=4)
    wd = g.rng.randint(ref.weekday() + 1, 6)
    p = g.plan(monday(ref.date()) + timedelta(days=wd), f"이번 주 {WD[wd]}요일")
    f = g.friend()
    c = Convo(f, ref)
    c.say(f, p.proposal())
    c.say(ME, "아 그날은 안 될 것 같아 ㅠㅠ 다음에 보자")
    g.add(c, [], ["tentative"])

    # "I'll think about it"
    ref = g.ref_time()
    target = ref.date() + timedelta(days=5)
    p = g.plan(target, f"{target.month}월 {target.day}일")
    f = g.friend()
    c = Convo(f, ref)
    c.say(f, p.proposal())
    c.say(ME, "음 생각해볼게! 그 주가 좀 바빠서")
    g.add(c, [], ["tentative"])


def no_plan(g: Gen) -> None:
    chats = [
        [("민지", "어제 본 영화 진짜 재밌었다"), (ME, "ㅋㅋ 그치 결말 미쳤어"), ("민지", "다음 편도 나오면 좋겠다")],
        [("준호", "오늘 회의 몇 시에 끝났어?"), (ME, "6시 반쯤? 완전 길었음"), ("준호", "고생했다 ㅠ")],
        [("서연", "지난주 토요일에 간 카페 이름 뭐였지?"), (ME, "성수동에 있는 거? 이름 기억 안 나 ㅋㅋ"), ("서연", "찾아보고 알려줘!")],
    ]
    for chat in chats:
        ref = g.ref_time()
        c = Convo(chat[0][0], ref)
        for who, text in chat:
            c.say(who, text)
        g.add(c, [], ["no_plan"])


def multiple(g: Gen) -> None:
    # two confirmed appointments with the same friend
    ref = g.ref_time(max_wd=3)
    f = g.friend()
    p1 = g.plan(ref.date() + timedelta(days=1), "내일")
    wd = 5  # Saturday
    p2 = g.plan(monday(ref.date()) + timedelta(days=wd), f"이번 주 {WD[wd]}요일")
    c = Convo(f, ref)
    c.say(f, p1.proposal())
    c.say(ME, g.accept())
    c.say(f, "그리고 " + p2.proposal())
    c.say(ME, "그것도 좋아!")
    g.add(c, [p1.event(f), p2.event(f)], ["multiple"], {"events": [p1.event(f)]})

    # one confirmed, one only floated
    ref = g.ref_time()
    f = g.friend()
    p1 = g.plan(ref.date() + timedelta(days=2), "모레")
    target2 = ref.date() + timedelta(days=10)
    p2 = g.plan(target2, f"{target2.month}월 {target2.day}일")
    c = Convo(f, ref)
    c.say(ME, p1.proposal())
    c.say(f, g.accept())
    c.say(f, "아 그리고 " + p2.proposal())
    c.say(ME, "그건 아직 모르겠어 확정되면 말해줄게")
    g.add(c, [p1.event(f)], ["multiple", "tentative"], {"events": [p1.event(f), p2.event(f)]})

    # group-style: two friends, one plan agreed by both
    ref = g.ref_time()
    f1, f2 = g.rng.sample(FRIENDS, 2)
    p = g.plan(ref.date() + timedelta(days=6), "")
    p.date_text = f"{p.target.month}월 {p.target.day}일"
    c = Convo(f1, ref)
    c.say(f1, p.proposal())
    c.say(f2, "좋아! 나도 갈게")
    c.say(ME, "나도 콜")
    ev = p.event(f1)
    ev["title"] = f"{f1}, {josa_wa(f2)} {p.activity[1]}"
    g.add(c, [ev], ["multiple"])


# ------------------------------------------------------------------ main


def build() -> Gen:
    g = Gen(SEED)
    absolute(g, 6)
    relative_day(g, 6)
    weekday(g, 8)
    changed(g)
    cancelled(g, 4)
    tentative(g)
    no_plan(g)
    multiple(g)
    return g


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    g = build()
    write_jsonl(HERE / "cases.jsonl", g.cases)

    reference = [{"id": c["id"], "output": c["expected"]} for c in g.cases]
    write_jsonl(HERE / "fixtures" / "reference_outputs.jsonl", reference)

    broken = []
    for c in g.cases:
        out = g.broken.get(c["id"], c["expected"])
        broken.append({"id": c["id"], "output": out})
    # one truncated response, as happens when a model hits max tokens
    broken[0]["output"] = '{"events": [{"title": "민지와 저'
    write_jsonl(HERE / "fixtures" / "broken_outputs.jsonl", broken)

    tags: dict[str, int] = {}
    for c in g.cases:
        for t in c["tags"]:
            tags[t] = tags.get(t, 0) + 1
    print(f"{len(g.cases)} cases, {sum(len(c['expected']['events']) for c in g.cases)} expected events, "
          f"{len(g.broken) + 1} broken outputs; tags: {tags}")


if __name__ == "__main__":
    main()
