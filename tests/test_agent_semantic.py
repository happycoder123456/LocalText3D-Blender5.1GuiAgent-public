"""Tests for the semantic layer: knowledge base, context tracker, abstraction, planner."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestKnowledge(unittest.TestCase):
    def test_hotkey_and_phrase_lookup(self):
        from agent import knowledge as kb

        self.assertEqual(kb.op_for_hotkey(["tab"], "OBJECT"), "object.editmode_toggle")
        self.assertEqual(kb.op_for_hotkey(["e"], "EDIT"), "mesh.extrude_region_move")
        self.assertIsNone(kb.op_for_hotkey(["e"], "OBJECT"))
        self.assertEqual(kb.op_for_hotkey(["ctrl", "r"], "EDIT"), "mesh.loopcut_slide")
        self.assertEqual(kb.op_for_hotkey(["3"], "EDIT"), "mesh.select_mode")
        self.assertEqual(kb.op_for_hotkey(["shift", "d"], "OBJECT"), "object.duplicate_move")
        self.assertEqual(kb.op_for_phrase("please bevel the edges"), "mesh.bevel")
        self.assertEqual(kb.op_for_phrase("thicken this wall"), "mesh.extrude_region_move")
        self.assertEqual(kb.op_for_search_text("Add Cube"), "mesh.primitive_cube_add")
        self.assertEqual(kb.normalize_idname("MESH_OT_extrude_region_move"), "mesh.extrude_region_move")
        self.assertTrue(kb.is_noise_op("view3d.rotate"))
        self.assertTrue(kb.is_noise_op("localtext3d.agent_record_stop"))
        self.assertEqual(kb.op_for_phrase("add a material"), "material.set")
        self.assertEqual(kb.op_for_phrase("material tab"), "material.set")
        self.assertEqual(kb.color_name_in_text("make me a red house"), "red")

    def test_expand_op_typed_params(self):
        from agent import knowledge as kb

        steps = kb.expand_op("mesh.extrude_region_move", {"distance": 1.5})
        kinds = [s["action"] for s in steps]
        self.assertEqual(steps[0]["keys"], ["e"])
        self.assertIn("type", kinds)
        typed = next(s for s in steps if s["action"] == "type")
        self.assertEqual(typed["text"], "1.5")
        self.assertEqual(steps[-1]["keys"], ["enter"])

        scale = kb.expand_op("transform.resize", {"factor": 2, "axis": "z"})
        self.assertEqual(scale[0]["keys"], ["s"])
        self.assertEqual([s for s in scale if s["action"] == "key"][1]["keys"], ["z"])
        self.assertEqual(next(s for s in scale if s["action"] == "type")["text"], "2")

        search = kb.expand_op("mesh.extrude_region_move", {"distance": 1.0}, variant=1)
        self.assertEqual(search[0]["keys"], ["f3"])
        self.assertTrue(any(s.get("text") == "Extrude Region" for s in search))

        cube = kb.expand_op("mesh.primitive_cube_add")
        self.assertTrue(any(s.get("text") == "Add Cube" for s in cube))
        self.assertEqual(kb.expand_op("mesh.select_mode", {"type": "FACE"})[0]["keys"], ["3"])
        self.assertEqual(kb.expand_op("object.subdivision_set", {"level": 2})[0]["keys"], ["ctrl", "2"])
        self.assertEqual(kb.expand_op("mesh.select_all", {"action": "DESELECT"})[0]["keys"], ["alt", "a"])

    def test_params_from_blender_props(self):
        from agent import knowledge as kb

        p = kb.params_from_props(
            "MESH_OT_extrude_region_move",
            {"MESH_OT_extrude_region": {}, "TRANSFORM_OT_translate": {"value": [0.0, 0.0, 1.2]}},
        )
        self.assertAlmostEqual(p["distance"], 1.2)
        s = kb.params_from_props("TRANSFORM_OT_resize", {"value": [2.0, 2.0, 2.0]})
        self.assertAlmostEqual(s["factor"], 2.0)
        self.assertEqual(s["axis"], "")
        t = kb.params_from_props("transform.translate", {"value": [0.0, 3.0, 0.0]})
        self.assertEqual(t["axis"], "y")
        self.assertAlmostEqual(t["distance"], 3.0)
        r = kb.params_from_props("transform.rotate", {"value": 1.5707963, "orient_axis": "Z"})
        self.assertAlmostEqual(r["angle"], 90.0, places=1)
        self.assertEqual(r["axis"], "z")


class TestContextTracker(unittest.TestCase):
    def test_update_ops_and_regions(self):
        from agent.context import ContextTracker

        tr = ContextTracker()
        tr.mark_session_start()
        tr.update(
            {
                "mode": "EDIT_MESH",
                "active": "Cube",
                "faces": 6,
                "sel_faces": 1,
                "window": {"w": 1920, "h": 1080},
                "viewport": {"x": 100, "y": 60, "w": 1400, "h": 900},
                "ops": [
                    {"op": "VIEW3D_OT_rotate", "props": {}},
                    {"op": "MESH_OT_extrude_region_move", "props": {"TRANSFORM_OT_translate": {"value": [0, 0, 0.7]}}},
                    {"op": "localtext3d.agent_record_stop", "props": {}},
                ],
            }
        )
        self.assertEqual(tr.latest()["mode"], "EDIT")
        ops = tr.session_ops()
        self.assertEqual([o["op"] for o in ops], ["mesh.extrude_region_move"])
        self.assertAlmostEqual(ops[0]["params"]["distance"], 0.7)
        self.assertEqual(tr.point_region(500, 400), "viewport")
        self.assertEqual(tr.point_region(1888, 661), "ui")
        cx, cy = tr.viewport_center_norm()
        self.assertAlmostEqual(cx, 800 / 1920, places=3)
        self.assertIn("EDIT", tr.summary())


class TestAbstraction(unittest.TestCase):
    def _recording(self):
        """Mimics the real 'Extrude the selected face' session: Tab, header click
        (face mode), viewport click, E + confirm click, Tab, then Stop in the N-panel."""
        t0 = time.time()
        events = [
            {"type": "key", "key": "tab", "t": t0 + 0.0},
            {"type": "click", "x": 238, "y": 46, "pressed": True, "coord_space": "window", "t": t0 + 1.0},
            {"type": "click", "x": 238, "y": 46, "pressed": False, "coord_space": "window", "t": t0 + 1.1},
            {"type": "click", "x": 1070, "y": 576, "pressed": True, "coord_space": "window", "t": t0 + 2.0},
            {"type": "click", "x": 1070, "y": 576, "pressed": False, "coord_space": "window", "t": t0 + 2.1},
            {"type": "key", "key": "e", "t": t0 + 3.0},
            {"type": "click", "x": 1067, "y": 782, "pressed": True, "coord_space": "window", "t": t0 + 4.0},
            {"type": "click", "x": 1067, "y": 782, "pressed": False, "coord_space": "window", "t": t0 + 4.1},
            {"type": "key", "key": "tab", "t": t0 + 5.0},
            {"type": "click", "x": 1888, "y": 661, "pressed": True, "coord_space": "window", "t": t0 + 6.0},
            {"type": "click", "x": 1888, "y": 661, "pressed": False, "coord_space": "window", "t": t0 + 6.1},
        ]
        return t0, events

    def test_infer_from_hotkeys_only(self):
        from agent.abstract import abstract_recording, infer_ops_from_actions
        from agent.recorder import events_to_actions

        _t0, events = self._recording()
        actions = events_to_actions(events)
        ops = infer_ops_from_actions(actions, start_mode="OBJECT", window_w=1920, window_h=1080)
        names = [o["op"] for o in ops]
        self.assertEqual(names[0], "object.editmode_toggle")
        self.assertIn("mesh.extrude_region_move", names)
        self.assertIn("view3d.select", names)
        # Header click (238,46) and N-panel click (1888,661) are not viewport clicks;
        # the extrude-confirm click is consumed by the extrude op.
        self.assertEqual(names.count("view3d.select"), 1)
        card = abstract_recording("Extrude the selected face", actions, window_w=1920, window_h=1080)
        # Trailing N-panel Stop click must not survive.
        self.assertNotEqual(card["ops"][-1]["op"], "view3d.select")
        self.assertIn("extrude", card["summary"])
        self.assertFalse(card["grounded"])

    def test_grounded_by_operator_history(self):
        from agent.abstract import abstract_recording
        from agent.context import ContextTracker
        from agent.recorder import events_to_actions

        t0, events = self._recording()
        tr = ContextTracker()
        tr._session_start = t0 - 0.5
        base = {"window": {"w": 1920, "h": 1080}, "viewport": {"x": 60, "y": 55, "w": 1500, "h": 980}, "active": "Cube"}

        def snap(t, **kw):
            payload = {**base, **kw}
            with mock.patch("agent.context.time.time", return_value=t):
                tr.update(payload)

        snap(t0 - 0.2, mode="OBJECT", faces=6)
        snap(t0 + 0.15, mode="EDIT_MESH", faces=6, ops=[{"op": "OBJECT_OT_editmode_toggle", "props": {}}])
        snap(t0 + 1.15, mode="EDIT_MESH", faces=6, ops=[{"op": "MESH_OT_select_mode", "props": {"type": "FACE"}}])
        snap(t0 + 2.2, mode="EDIT_MESH", faces=6, sel_faces=1)
        snap(
            t0 + 4.2,
            mode="EDIT_MESH",
            faces=10,
            sel_faces=1,
            ops=[{"op": "MESH_OT_extrude_region_move", "props": {"TRANSFORM_OT_translate": {"value": [0, 0, 1.25]}}}],
        )
        snap(t0 + 5.15, mode="OBJECT", faces=10, ops=[{"op": "OBJECT_OT_editmode_toggle", "props": {}}])
        snap(t0 + 6.15, mode="OBJECT", faces=10)

        actions = events_to_actions(events)
        card = abstract_recording(
            "Extrude the selected face", actions, tracker=tr, input_events=events, window_w=1920, window_h=1080
        )
        names = [o["op"] for o in card["ops"]]
        self.assertTrue(card["grounded"])
        self.assertEqual(names[0], "object.editmode_toggle")
        self.assertIn("mesh.select_mode", names)
        self.assertIn("view3d.select", names)
        ext = next(o for o in card["ops"] if o["op"] == "mesh.extrude_region_move")
        self.assertAlmostEqual(ext["params"]["distance"], 1.25)
        self.assertEqual(names[-1], "object.editmode_toggle")  # Stop click dropped
        self.assertEqual(card["start_mode"], "OBJECT")
        self.assertEqual(card["effects"]["faces_delta"], 4)
        self.assertIn("1.25", card["summary"])

    def test_ops_to_steps_inserts_mode_switch_and_tags(self):
        from agent.abstract import new_op, ops_to_steps

        ops = [new_op("mesh.extrude_region_move", {"distance": 2.0})]
        steps = ops_to_steps(ops, start_mode="OBJECT")
        self.assertEqual(steps[0]["keys"], ["tab"])
        self.assertTrue(steps[0].get("auto_mode"))
        lasts = [s for s in steps if s.get("sem_last")]
        self.assertEqual(len(lasts), 1)
        self.assertEqual(lasts[0]["keys"], ["enter"])
        self.assertTrue(all(s.get("sem_i") == 0 for s in steps))

        # Already in Edit mode: no Tab.
        steps2 = ops_to_steps(ops, start_mode="EDIT")
        self.assertEqual(steps2[0]["keys"], ["e"])

    def test_judge_op_with_context(self):
        from agent.judge import judge_op

        before = {"mode": "EDIT", "faces": 6, "verts": 8}
        after = {"mode": "EDIT", "faces": 10, "verts": 12}
        r, note = judge_op("mesh.extrude_region_move", {}, before, after, [{"op": "mesh.extrude_region_move"}])
        self.assertGreater(r, 0.5)
        r2, _ = judge_op("mesh.extrude_region_move", {}, before, before, [])
        self.assertLess(r2, 0)
        r3, _ = judge_op("object.editmode_toggle", {}, {"mode": "OBJECT"}, {"mode": "EDIT"}, [])
        self.assertGreater(r3, 0)
        r4, _ = judge_op("mesh.bevel", {}, None, None, [])
        self.assertIsNone(r4)


class TestPlanner(unittest.TestCase):
    def _skills(self):
        from agent.abstract import make_skill_card, new_op

        card = make_skill_card(
            "Extrude the selected face",
            [
                new_op("object.editmode_toggle"),
                new_op("mesh.select_mode", {"type": "FACE"}),
                new_op("view3d.select", pointer={"x_norm": 0.55, "y_norm": 0.53}),
                new_op("mesh.extrude_region_move", {"distance": 1.25}),
                new_op("object.editmode_toggle"),
            ],
            start_mode="OBJECT",
        )
        return [card]

    def test_variation_of_recorded_skill(self):
        from agent.planner import plan_with_rules

        plan = plan_with_rules("extrude the selected face by 3", self._skills())
        self.assertEqual(plan.source, "rules")
        self.assertIn("Extrude the selected face", plan.used_skills)
        ext = next(o for o in plan.ops if o["op"] == "mesh.extrude_region_move")
        self.assertAlmostEqual(ext["params"]["distance"], 3.0)
        # The learned structure is kept (select face, extrude, back to object mode).
        self.assertEqual(plan.ops[0]["op"], "object.editmode_toggle")
        self.assertFalse(plan.unresolved)

    def test_compose_new_goal_from_knowledge(self):
        from agent.planner import enrich_plan_for_start, plan_with_rules

        plan = plan_with_rules("add a cube of size 2, then extrude by 0.5, then bevel the edges", [])
        names = [o["op"] for o in plan.ops]
        self.assertEqual(names[0], "mesh.primitive_cube_add")
        self.assertEqual(names[1], "transform.resize")
        self.assertAlmostEqual(plan.ops[1]["params"]["factor"], 2.0)
        self.assertIn("mesh.extrude_region_move", names)
        self.assertAlmostEqual(next(o for o in plan.ops if o["op"] == "mesh.extrude_region_move")["params"]["distance"], 0.5)
        self.assertIn("mesh.bevel", names)
        self.assertFalse(plan.unresolved)
        enriched = enrich_plan_for_start(plan, {"mode": "OBJECT", "sel_faces": 0})
        names2 = [o["op"] for o in enriched.ops]
        self.assertIn("mesh.select_all", names2)
        self.assertLess(names2.index("mesh.select_all"), names2.index("mesh.extrude_region_move"))

    def test_repeat_and_axis_and_unresolved(self):
        from agent.planner import plan_with_rules

        plan = plan_with_rules("scale it by 2 on z, then move 3 along x twice, then make it look like a castle", [])
        scale = plan.ops[0]
        self.assertEqual(scale["op"], "transform.resize")
        self.assertAlmostEqual(scale["params"]["factor"], 2.0)
        self.assertEqual(scale["params"]["axis"], "z")
        moves = [o for o in plan.ops if o["op"] == "transform.translate"]
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0]["params"]["axis"], "x")
        self.assertTrue(plan.unresolved)
        self.assertIn("castle", plan.unresolved[0])

    def test_does_not_add_primitive_when_just_mentioned(self):
        from agent.planner import plan_with_rules

        plan = plan_with_rules("scale the cube by 2", [])
        self.assertEqual([o["op"] for o in plan.ops], ["transform.resize"])
        plan2 = plan_with_rules("3 cubes", [])
        self.assertEqual([o["op"] for o in plan2.ops], ["mesh.primitive_cube_add"] * 3)

    def test_llm_plan_validation(self):
        from agent.planner import _validate_llm_plan

        plan = _validate_llm_plan(
            {
                "plan": [
                    {"skill": "extrude the selected face", "params": {"distance": 4}},
                    {"op": "mesh.bevel", "params": {"offset": "0.2", "segments": 3, "bogus": 1}},
                    {"op": "made.up_op", "params": {}},
                    {"op": "MESH_OT_inset", "params": {"thickness": 0.1}},
                ],
                "why": "test",
            },
            self._skills(),
        )
        names = [o["op"] for o in plan.ops]
        self.assertIn("mesh.extrude_region_move", names)
        self.assertAlmostEqual(next(o for o in plan.ops if o["op"] == "mesh.extrude_region_move")["params"]["distance"], 4.0)
        bevel = next(o for o in plan.ops if o["op"] == "mesh.bevel")
        self.assertEqual(bevel["params"], {"offset": 0.2, "segments": 3})
        self.assertNotIn("made.up_op", names)
        self.assertIn("mesh.inset", names)

    def test_make_plan_falls_back_to_llm_only_when_needed(self):
        from agent import planner

        calls = []

        def fake_llm(goal, skills, *, model, context=None, rules_hint=None):
            calls.append(goal)
            return planner.Plan(
                ops=[
                    planner.new_op("mesh.primitive_cube_add", {}),
                    planner.new_op("transform.resize", {"factor": 1.4, "axis": "z"}),
                    planner.new_op("mesh.inset", {"thickness": 0.1}),
                    planner.new_op("mesh.extrude_region_move", {"distance": 0.4}),
                    planner.new_op("mesh.bevel", {"offset": 0.05}),
                ],
                source="llm",
                why="ok",
            )

        with mock.patch.object(planner, "plan_with_llm", side_effect=fake_llm):
            p1 = planner.make_plan("extrude by 2", skills=[], model="llama3.1")
            self.assertEqual(p1.source, "rules")
            self.assertEqual(calls, [])
            p2 = planner.make_plan("make a castle", skills=[], model="llama3.1")
            self.assertEqual(p2.source, "llm")
            self.assertEqual(calls, ["make a castle"])

    def test_car_and_house_use_techniques_not_same_box(self):
        from agent.abstract import new_op
        from agent.planner import is_shallow_box_ops, make_plan

        skills = [
            {
                "goal": "extrude",
                "origin": "video",
                "ops": [new_op("mesh.extrude_region_move", {"distance": 1.0})],
                "aliases": ["thicken"],
            },
            {
                "goal": "inset",
                "origin": "video",
                "ops": [new_op("mesh.inset", {"thickness": 0.1})],
            },
            {
                "goal": "bevel",
                "origin": "video",
                "ops": [new_op("mesh.bevel", {"offset": 0.05})],
            },
            {
                "goal": "loop cut",
                "origin": "video",
                "ops": [new_op("mesh.loopcut_slide", {"number_cuts": 1})],
            },
            # Poison: click-spam recording that used to hijack "make a house".
            {
                "goal": "make me a advanced house",
                "origin": "recorded",
                "ops": [new_op("view3d.select")] * 10
                + [
                    new_op("mesh.primitive_cube_add"),
                    new_op("mesh.bevel"),
                    new_op("mesh.extrude_region_move"),
                ],
            },
            {
                "goal": "make me a simple car",
                "origin": "planned",
                "ops": [
                    new_op("mesh.primitive_cube_add"),
                    new_op("transform.resize", {"factor": 2.0, "axis": "z"}),
                    new_op("mesh.inset", {"thickness": 0.1}),
                ],
            },
        ]
        car = make_plan("make me a simple car", skills=skills, model="", allow_llm=False)
        house = make_plan("make me a house", skills=skills, model="", allow_llm=False)
        self.assertEqual(car.source, "compose")
        self.assertEqual(house.source, "compose")
        self.assertNotEqual([o["op"] for o in car.ops], [o["op"] for o in house.ops])
        self.assertTrue(any(o["op"] == "mesh.primitive_cylinder_add" for o in car.ops))
        self.assertFalse(any(o["op"] == "mesh.primitive_cylinder_add" for o in house.ops))
        self.assertTrue(any(o["op"] == "mesh.primitive_cube_add" for o in house.ops))
        # House should include a separate roof slab (second cube), not click spam.
        self.assertGreaterEqual(sum(1 for o in house.ops if o["op"] == "mesh.primitive_cube_add"), 2)
        self.assertFalse(any(o["op"] == "view3d.select" for o in house.ops))
        self.assertFalse(is_shallow_box_ops(car.ops))
        self.assertFalse(is_shallow_box_ops(house.ops))

    def test_search_only_house_skill_does_not_hijack_plan(self):
        from agent.abstract import new_op
        from agent.planner import make_plan

        poison = {
            "goal": "make me an advanced house",
            "origin": "video",
            "ops": [new_op("search", {"text": "Make Me An Advanced House"})],
            "summary": "Technique inferred from bad seed",
        }
        plan = make_plan(
            "make me an advanced house",
            skills=[poison, {"goal": "extrude", "origin": "video", "ops": [new_op("mesh.extrude_region_move")]}],
            model="",
            allow_llm=False,
        )
        self.assertEqual(plan.source, "compose")
        self.assertGreaterEqual(len(plan.ops), 14)
        self.assertFalse(any(o.get("op") == "search" for o in plan.ops))
        self.assertTrue(any(o.get("op") == "mesh.primitive_cube_add" for o in plan.ops))

    def test_plan_length_scales_with_goal_complexity(self):
        from agent.planner import compose_object_plan, make_plan, plan_budget

        simple = make_plan("add a cube", skills=[], model="", allow_llm=False)
        house = compose_object_plan("make me a house", [])
        advanced = compose_object_plan(
            "make me a very detailed house with windows doors roof chimney garage balcony",
            [],
        )
        car = compose_object_plan("make me a car", [])
        self.assertLessEqual(len(simple.ops), 4)
        self.assertGreaterEqual(len(house.ops), 14)
        self.assertGreater(len(advanced.ops), len(house.ops))
        self.assertGreaterEqual(len(car.ops), 20)
        self.assertGreater(plan_budget("make a realistic car"), plan_budget("add a cube"))
        self.assertGreaterEqual(plan_budget("make an advanced house with windows and a garage"), 100)
        self.assertGreaterEqual(plan_budget("make me a very detailed house with windows doors roof"), 120)
        # Payload exposes full step list + count (UI used to truncate to 12–14).
        payload = advanced.to_payload()
        self.assertEqual(payload["n_ops"], len(advanced.ops))
        self.assertEqual(len(payload["steps"]), len(advanced.ops))

    def test_red_house_uses_material_and_learned_skills(self):
        from agent.abstract import new_op
        from agent.planner import make_plan

        skills = [
            {
                "goal": "bevel",
                "origin": "video",
                "ops": [new_op("mesh.bevel", {"offset": 0.05})],
            },
            {
                "goal": "solidify",
                "origin": "video",
                "ops": [new_op("mesh.solidify", {"thickness": 0.08})],
            },
            {
                "goal": "material",
                "origin": "video",
                "ops": [new_op("material.set", {"color": "red"})],
            },
        ]
        red = make_plan("make me a red house", skills=skills, model="", allow_llm=False)
        plain = make_plan("make me a house", skills=skills, model="", allow_llm=False)
        red_ops = [o["op"] for o in red.ops]
        self.assertIn("material.set", red_ops)
        mat = next(o for o in red.ops if o["op"] == "material.set")
        self.assertEqual(str(mat["params"].get("color") or "").lower(), "red")
        self.assertTrue(any("red" in str(s).lower() for s in red.used_skills))
        self.assertNotIn("material.set", [o["op"] for o in plain.ops])
        self.assertGreater(red_ops.index("material.set"), red_ops.index("mesh.primitive_cube_add"))

    def test_golden_gate_is_a_bridge_not_a_house(self):
        from agent.planner import compose_object_plan, make_plan, named_object, object_kind

        self.assertEqual(named_object("make me a red golden gate bridge"), "golden_gate")
        self.assertEqual(object_kind("make me a red golden gate bridge"), "golden_gate")
        house = compose_object_plan("make me a house", [])
        bridge = make_plan("make me a red golden gate bridge", skills=[], model="", allow_llm=False)
        self.assertEqual(bridge.source, "compose")
        house_ops = [o["op"] for o in house.ops]
        bridge_ops = [o["op"] for o in bridge.ops]
        self.assertNotEqual(house_ops, bridge_ops)
        self.assertGreaterEqual(sum(1 for op in bridge_ops if op == "mesh.primitive_cube_add"), 4)
        self.assertGreaterEqual(sum(1 for op in bridge_ops if op == "mesh.primitive_cylinder_add"), 4)
        self.assertTrue(
            any(
                o["op"] == "transform.resize"
                and str((o.get("params") or {}).get("axis") or "") == "x"
                and float((o.get("params") or {}).get("factor") or 0) >= 3.0
                for o in bridge.ops
            )
        )
        self.assertIn("material.set", bridge_ops)
        self.assertEqual(str(next(o for o in bridge.ops if o["op"] == "material.set")["params"].get("color")), "red")
        self.assertIn("suspension", (bridge.why or "").lower())

    def test_llm_house_plan_is_rejected_for_golden_gate(self):
        from agent import planner

        def fake_llm(goal, skills, *, model, context=None, rules_hint=None):
            return planner.Plan(
                ops=[
                    planner.new_op("mesh.primitive_cube_add", {}),
                    planner.new_op("transform.resize", {"factor": 1.4, "axis": "z"}),
                    planner.new_op("mesh.inset", {"thickness": 0.1}),
                    planner.new_op("mesh.extrude_region_move", {"distance": 0.4}),
                    planner.new_op("mesh.bevel", {"offset": 0.05}),
                ],
                source="llm",
                why="house",
            )

        with mock.patch.object(planner, "plan_with_llm", side_effect=fake_llm):
            plan = planner.make_plan("make me a golden gate bridge", skills=[], model="llama3.2-vision")
        self.assertEqual(plan.source, "compose")
        self.assertGreaterEqual(sum(1 for o in plan.ops if o["op"] == "mesh.primitive_cylinder_add"), 4)

    def test_any_named_object_is_not_a_house(self):
        from agent.planner import compose_object_plan, is_complex_object_goal, make_plan, structure_family

        self.assertFalse(is_complex_object_goal("add a cube"))
        self.assertFalse(is_complex_object_goal("extrude by 2"))
        self.assertTrue(is_complex_object_goal("make me a piano"))
        self.assertTrue(is_complex_object_goal("make me a spaceship"))
        self.assertEqual(structure_family("make me a piano"), "instrument")
        self.assertEqual(structure_family("make me a spaceship"), "aircraft")
        house = compose_object_plan("make me a house", [])
        piano = make_plan("make me a piano", skills=[], model="", allow_llm=False)
        ship = make_plan("make me a spaceship", skills=[], model="", allow_llm=False)
        self.assertNotEqual([o["op"] for o in house.ops], [o["op"] for o in piano.ops])
        self.assertNotEqual([o["op"] for o in house.ops], [o["op"] for o in ship.ops])
        self.assertGreaterEqual(sum(1 for o in piano.ops if o["op"].startswith("mesh.primitive_")), 3)
        self.assertGreaterEqual(sum(1 for o in ship.ops if o["op"].startswith("mesh.primitive_")), 3)
        self.assertIn("instrument", (piano.why or "").lower())

    def test_llm_house_plan_is_rejected_for_any_object(self):
        from agent import planner

        def fake_llm(goal, skills, *, model, context=None, rules_hint=None):
            return planner.Plan(
                ops=[
                    planner.new_op("mesh.primitive_cube_add", {}),
                    planner.new_op("transform.resize", {"factor": 1.4, "axis": "z"}),
                    planner.new_op("mesh.inset", {"thickness": 0.1}),
                    planner.new_op("mesh.extrude_region_move", {"distance": 0.4}),
                    planner.new_op("mesh.bevel", {"offset": 0.05}),
                ],
                source="llm",
                why="house",
            )

        with mock.patch.object(planner, "plan_with_llm", side_effect=fake_llm):
            plan = planner.make_plan("make me a piano", skills=[], model="llama3.2-vision")
        self.assertEqual(plan.source, "compose")
        self.assertIn("instrument", (plan.why or "").lower())
        self.assertGreaterEqual(sum(1 for o in plan.ops if o["op"].startswith("mesh.primitive_")), 3)

    def test_plans_leave_layout_for_modeling_and_specialty_workspaces(self):
        from agent import knowledge as kb
        from agent.abstract import new_op
        from agent.planner import (
            Plan,
            desired_object_mode,
            desired_workspace,
            ensure_workspace_ops,
            make_plan,
        )

        # Normal object goals do not force a workspace — Layout/Modeling are fine.
        self.assertEqual(desired_workspace("make me a piano"), "")
        self.assertEqual(desired_workspace("unwrap the UV seams"), "UV Editing")
        self.assertEqual(desired_workspace("texture paint the car"), "Texture Paint")
        self.assertEqual(desired_workspace("sculpt the character"), "Sculpting")
        self.assertEqual(desired_object_mode("texture paint the car"), "TEXTURE_PAINT")
        self.assertEqual(desired_object_mode("sculpt the character"), "SCULPT")

        stay = make_plan(
            "make me a piano",
            skills=[],
            model="",
            allow_llm=False,
            context={"workspace": "Layout", "mode": "OBJECT"},
        )
        self.assertNotEqual(stay.ops[0]["op"], "workspace.set")

        leave_uv = ensure_workspace_ops(
            Plan(ops=[new_op("mesh.primitive_cube_add")], source="test"),
            "make me a piano",
            {"workspace": "UV Editing", "mode": "EDIT"},
        )
        self.assertEqual(leave_uv.ops[0]["op"], "workspace.set")
        self.assertEqual(leave_uv.ops[0]["params"].get("name"), "Modeling")

        paint = ensure_workspace_ops(
            Plan(ops=[new_op("mesh.select_all", {"action": "SELECT"})], source="test"),
            "texture paint the car",
            {"workspace": "Layout", "mode": "OBJECT"},
        )
        self.assertEqual(paint.ops[0]["op"], "workspace.set")
        self.assertEqual(paint.ops[0]["params"].get("name"), "Texture Paint")
        self.assertEqual(paint.ops[1]["op"], "object.mode_set")
        self.assertEqual(paint.ops[1]["params"].get("mode"), "TEXTURE_PAINT")

        already = ensure_workspace_ops(
            Plan(ops=[new_op("mesh.select_all", {"action": "SELECT"})], source="test"),
            "texture paint the car",
            {"workspace": "Texture Paint", "mode": "TEXTURE_PAINT", "raw_mode": "TEXTURE_PAINT"},
        )
        self.assertNotEqual(already.ops[0]["op"], "workspace.set")
        self.assertNotEqual(already.ops[0]["op"], "object.mode_set")

        steps = kb.expand_op("workspace.set", {"name": "Modeling"})
        self.assertEqual(steps[0]["action"], "blender_apply")
        self.assertEqual(steps[0]["apply"]["op"], "workspace.set")

    def test_any_object_uses_learned_techniques(self):
        """Video/record teach HOW; only goal-relevant techniques are grafted (not every card)."""
        from agent.abstract import new_op
        from agent.planner import is_technique_skill, make_plan

        skills = [
            {
                "goal": "bevel edges",
                "origin": "video",
                "ops": [new_op("mesh.bevel", {"offset": 0.04, "segments": 2})],
            },
            {
                "goal": "extrude",
                "origin": "concept",
                "ops": [new_op("mesh.extrude_region_move", {"distance": 0.25})],
            },
            {
                "goal": "solidify thickness",
                "origin": "video",
                "ops": [new_op("mesh.solidify", {"thickness": 0.06})],
            },
            {
                "goal": "make me a house",
                "origin": "planned",
                "ops": [new_op("mesh.primitive_cube_add"), new_op("mesh.extrude_region_move", {"distance": 1})],
            },
        ]
        self.assertTrue(is_technique_skill(skills[0]))
        self.assertFalse(is_technique_skill(skills[3]))
        piano = make_plan("make me a piano", skills=skills, model="", allow_llm=False)
        detailed = make_plan("make me a piano with bevel and extrude", skills=skills, model="", allow_llm=False)
        house = make_plan("make me a house", skills=skills, model="", allow_llm=False)
        self.assertGreaterEqual(sum(1 for o in piano.ops if o["op"].startswith("mesh.primitive_")), 3)
        # Instrument family hints include bevel — may graft; do not dump every technique.
        self.assertNotIn("make me a house", " ".join(str(s).lower() for s in piano.used_skills))
        self.assertNotEqual([o["op"] for o in house.ops], [o["op"] for o in piano.ops])
        det_ops = [o["op"] for o in detailed.ops]
        self.assertIn("mesh.bevel", det_ops)
        self.assertIn("mesh.extrude_region_move", det_ops)
        used = " ".join(str(s).lower() for s in detailed.used_skills)
        self.assertIn("bevel", used)
        self.assertIn("extrude", used)
        self.assertNotIn("make me a house", used)

    def test_expand_material_set_is_blender_apply(self):
        from agent import knowledge as kb

        steps = kb.expand_op("material.set", {"color": "blue", "name": "Blue"})
        self.assertEqual(steps[0]["action"], "blender_apply")
        self.assertEqual(steps[0]["apply"]["op"], "material.set")
        self.assertAlmostEqual(steps[0]["apply"]["color"][2], 0.85, places=2)


class TestLoopSemantic(unittest.TestCase):
    def test_stop_record_attaches_semantics_and_run_varies(self):
        from agent.loop import AgentLoop

        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(dataset_root=Path(tmp))
            t0 = time.time() - 10.0
            loop.context._session_start = t0 - 0.5
            base = {"window": {"w": 1600, "h": 900}, "viewport": {"x": 50, "y": 50, "w": 1200, "h": 800}}

            def snap(t, **kw):
                with mock.patch("agent.context.time.time", return_value=t):
                    loop.context.update({**base, **kw})

            snap(t0 - 0.2, mode="OBJECT", faces=6)
            snap(t0 + 0.2, mode="EDIT_MESH", faces=6, ops=[{"op": "OBJECT_OT_editmode_toggle", "props": {}}])
            snap(t0 + 0.7, mode="EDIT_MESH", faces=6, sel_faces=1)
            snap(t0 + 1.3, mode="EDIT_MESH", faces=10, sel_faces=1, ops=[{"op": "MESH_OT_extrude_region_move", "props": {"TRANSFORM_OT_translate": {"value": [0, 0, 0.8]}}}])
            loop.recorder._session_id = "sem1"
            loop.recorder._session_started_at = t0
            loop.recorder._session_goal = "extrude face"
            loop.recorder._session_events = [
                {"type": "key", "key": "tab", "t": t0 + 0.1},
                {"type": "click", "x": 600, "y": 400, "pressed": True, "coord_space": "window", "t": t0 + 0.5},
                {"type": "click", "x": 600, "y": 400, "pressed": False, "coord_space": "window", "t": t0 + 0.55},
                {"type": "key", "key": "e", "t": t0 + 1.0},
            ]
            loop.recorder._running = False

            def _fake_stop(goal: str = ""):
                loop.recorder._running = False
                if goal:
                    loop.recorder._session_goal = goal
                loop.recorder._write_last_session()

            loop.recorder.stop = _fake_stop  # type: ignore[method-assign]
            with mock.patch("agent.loop.find_blender_window", return_value=None):
                loop.stop_record(goal="extrude face")
            self.assertIn("Technique", loop._detail)
            cards = loop.memory.semantic_skills()
            self.assertEqual(cards[0]["goal"], "extrude face")
            self.assertTrue(cards[0]["grounded"])
            ext = next(o for o in cards[0]["ops"] if o["op"] == "mesh.extrude_region_move")
            self.assertAlmostEqual(ext["params"]["distance"], 0.8)

            # Ask for a variation: same technique, different number.
            def _fake_run(max_steps: int) -> None:
                loop._running = False
                loop._status = "Done"

            loop._run = _fake_run  # type: ignore[method-assign]
            with mock.patch("agent.loop.resolve_planner_model", return_value=""):
                loop.start_agent(goal="extrude face by 2.5", model="none", run_mode="learn")
            if loop._thread is not None:
                loop._thread.join(timeout=2.0)
            self.assertIsNotNone(loop._plan)
            self.assertEqual(loop._mode, "master")
            typed = [s for s in loop._skill_steps if s.get("action") == "type"]
            self.assertTrue(any(s.get("text") == "2.5" for s in typed))
            self.assertIn("2.5", loop._detail)

    def test_semantic_step_judged_and_retried(self):
        from agent.abstract import new_op, ops_to_steps
        from agent.loop import AgentLoop
        from agent.planner import Plan

        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(dataset_root=Path(tmp))
            loop._goal = "bevel"
            loop._plan = Plan(ops=[new_op("mesh.bevel", {"offset": 0.1})], source="rules")
            steps = ops_to_steps(loop._plan.ops, start_mode="EDIT")
            loop._skill_steps = list(steps)
            loop._skill_steps_live = [dict(s) for s in steps]
            loop._skill_step_i = len(steps)
            loop.context.update({"mode": "EDIT_MESH", "faces": 6})
            loop._begin_semantic_step(steps[0], None)
            # Blender reports nothing changed → miss → retry via F3 search inserted.
            loop.context.update({"mode": "EDIT_MESH", "faces": 6})
            last = next(s for s in steps if s.get("sem_last"))
            loop._finish_semantic_step(last, None, "")
            self.assertLess(loop._last_reward, 0)
            inserted = loop._skill_steps[len(steps):]
            self.assertTrue(inserted)
            self.assertEqual(inserted[0]["keys"], ["f3"])
            self.assertEqual(loop.memory.preferred_variant("mesh.bevel"), 1)


if __name__ == "__main__":
    unittest.main()
