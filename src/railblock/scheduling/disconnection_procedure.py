"""Grounds the multi-department combination mechanism in the real
Southern Railway disconnection/reconnection procedure -- SR 3.51.6 &
App. XIII, Southern Railway ASM training guide (docs/1429770763529-Pro
ASM study material.pdf, p.27-28). This is a labelling/documentation
improvement, not new solver logic.

Real procedure this cites:
  - "For works involving disconnection for more than one hour, a
    Disconnection schedule jointly signed by Sr.DSTE, Sr.DOM, Sr.DEN &
    Sr.DEE/TRD shall be issued" -- i.e. Signalling, Operations,
    Engineering, and Traction sign-off, exactly the four departments this
    system already coordinates across (see railblock.synthetic.
    maintenance_tasks.DEPARTMENTS + Pour et al.-adapted combination logic
    in scheduling/model.py). This is citable evidence that the
    combination mechanism isn't an invented convenience -- it mirrors a
    real joint sign-off requirement.
  - "The SI shall issue reconnection notice only after he receives Track
    fit Certificate. The SM shall test the signals, points... jointly with
    SI before accepting reconnection." -- reconnection is a verified,
    separate event from the scheduled window simply ending, which is why
    `completion_verified` (threaded through the schedule output in
    model.py/orchestrator.py) defaults to False rather than being implied
    by the window's end time.
"""

from __future__ import annotations

REQUIRED_SIGNOFF_ROLES_JOINT_SCHEDULE = ["Sr.DSTE", "Sr.DOM", "Sr.DEN", "Sr.DEE/TRD"]
