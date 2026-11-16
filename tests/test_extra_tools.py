"""Which task-specific extra tools each interface gets: the default tool lists are unchanged; point_tool exists on every
interface; pull only on the force interface."""
from forceharness.agent.tools import tool_schemas


def names(iface, extra=()):
    return [t["name"] for t in tool_schemas(iface, True, extra)]


def test_default_tool_lists_unchanged():
    for iface in ("pose", "guard", "force"):
        n = names(iface)
        assert "point_tool" not in n and "pull" not in n


def test_point_tool_all_interfaces_pull_force_only():
    extra = ("point_tool", "pull")
    assert "point_tool" in names("pose", extra) and "pull" not in names("pose", extra)
    assert "point_tool" in names("guard", extra) and "pull" not in names("guard", extra)
    f = names("force", extra)
    assert "point_tool" in f and "pull" in f and f[-1] == "done"


def test_push_click_flag_only_changes_push_when_enabled():
    base = {t["name"]: t for t in tool_schemas("force", True)}
    on = {t["name"]: t for t in tool_schemas("force", True, ("point_tool", "pull", "push_click"))}
    assert "push_click" not in on
    assert "stop_on_resistance_drop" not in base["push"]["parameters"]["properties"]
    assert "stop_on_resistance_drop" in on["push"]["parameters"]["properties"]
    assert on["push"]["description"].startswith(base["push"]["description"])
