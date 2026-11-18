"""Unit tests for the water-filling budget allocation algorithm."""

from context_assembler.allocation.water_fill import water_fill
from context_assembler.domain.budget import SlotPolicy


def test_water_fill_respects_minimum_floors() -> None:
    policy = {
        "rag": SlotPolicy("rag", min_tokens=100, max_tokens=1000, weight=1),
        "history": SlotPolicy("history", min_tokens=200, max_tokens=1000, weight=1),
    }
    demand = {"rag": 500, "history": 500}
    order = ["rag", "history"]

    # Budget is just enough to cover floors (100 + 200 = 300)
    alloc = water_fill(budget=300, demand=demand, policy=policy, order=order)
    assert alloc["rag"] == 100
    assert alloc["history"] == 200


def test_water_fill_respects_maximum_ceilings() -> None:
    policy = {
        "rag": SlotPolicy("rag", min_tokens=50, max_tokens=300, weight=1),
        "history": SlotPolicy("history", min_tokens=50, max_tokens=500, weight=1),
    }
    demand = {"rag": 1000, "history": 1000}
    order = ["rag", "history"]

    # Budget is 2000, but rag is capped at 300 and history at 500
    alloc = water_fill(budget=2000, demand=demand, policy=policy, order=order)
    assert alloc["rag"] == 300
    assert alloc["history"] == 500


def test_water_fill_distributes_surplus_proportionally_by_weight() -> None:
    policy = {
        "rag": SlotPolicy("rag", min_tokens=100, max_tokens=2000, weight=3),
        "history": SlotPolicy("history", min_tokens=100, max_tokens=2000, weight=1),
    }
    demand = {"rag": 2000, "history": 2000}
    order = ["rag", "history"]

    # Floors take 200. Surplus is 400.
    # rag weight 3 -> receives 300 surplus -> total 400
    # history weight 1 -> receives 100 surplus -> total 200
    alloc = water_fill(budget=600, demand=demand, policy=policy, order=order)
    assert alloc["rag"] == 400
    assert alloc["history"] == 200


def test_water_fill_zero_budget() -> None:
    policy = {
        "rag": SlotPolicy("rag", min_tokens=100, max_tokens=1000, weight=1),
    }
    alloc = water_fill(budget=0, demand={"rag": 500}, policy=policy, order=["rag"])
    assert alloc["rag"] == 0

