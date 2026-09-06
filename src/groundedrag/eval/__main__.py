"""允许 ``python -m groundedrag.eval`` 直接运行评测。"""

from groundedrag.eval.runner import main

raise SystemExit(main())  # pragma: no cover —— python -m 入口，由集成测试覆盖
