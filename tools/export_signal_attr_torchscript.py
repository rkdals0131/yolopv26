from __future__ import annotations

from pathlib import Path
import site

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from tools.model_export.signal_attr_torchscript import main


if __name__ == "__main__":
    main()
