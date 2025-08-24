# Issue: `TypeError: non-default argument 'model' follows default argument`

## Date
2025-08-24

## Description
The test suite is failing during test collection with a `TypeError: non-default argument 'model' follows default argument`. This error seems to be related to the interaction between Python's `dataclasses` and `abc` modules.

The error originates from the `EuropeanOption` dataclass in `src/options.py`.

## What I've tried
1.  **Removing the `@dataclass` decorator:** This fixes the collection error, but introduces a new `TypeError: Can't instantiate abstract class ... without an implementation for abstract method 'maturity'`. This error is misleading, as `maturity` is not an abstract method.
2.  **Removing `ABC` from `Instrument`:** This did not solve the issue.
3.  **Reordering fields:** The field with a default value is already the last one.
4.  **Explicitly setting `abstractmethod=False`:** This is not a feature of dataclasses.
5.  **Adding `ABC` to `EuropeanOption`:** This did not solve the issue.

## Hypothesis
There is a subtle interaction between `@dataclass` and `ABC` that is causing this issue. The error message is likely misleading. The problem might be related to how dataclasses handle inheritance and abstract methods.
