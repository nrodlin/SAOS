# Logging

All SAOS modules share one logging queue and one listener thread per process. Scripts create the logger with
`LoggingHelper(level).logger` and pass it to every module (`logger=...`); a module created without a logger uses
`get_logger()`, which returns the same shared root logger.

```python
import logging
from SAOS.LoggingHelper import LoggingHelper
from SAOS.Telescope import Telescope

log = LoggingHelper(logging.INFO)
tel = Telescope(diameter=1.5, resolution=64, logger=log.logger)
```

::: SAOS.LoggingHelper.get_logger

::: SAOS.LoggingHelper.LoggingHelper
    options:
      show_root_heading: true
      members: true
