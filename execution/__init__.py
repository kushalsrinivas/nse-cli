"""Paper execution for the order-block system.

There is no live order path in this package, by design: nothing here
imports `kiteconnect`, and tests/test_ob_paper.py asserts it. The broker
mirrors KiteConnect.place_order's signature so a future live adapter is a
mechanical swap — building one is a separate, deliberate decision.
"""
