from pathlib import Path
import site

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from tools.check_env import main


if __name__ == "__main__":
    raise SystemExit(main())
