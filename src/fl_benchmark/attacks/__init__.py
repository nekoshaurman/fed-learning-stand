"""Data-poisoning attack components for controlled FL simulations."""

from fl_benchmark.attacks.label_flip import LabelFlipPlan, build_label_flip_plan, make_client_dataset

__all__ = ["LabelFlipPlan", "build_label_flip_plan", "make_client_dataset"]
