from pathlib import Path
import site

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from tools.pv26_train.cli import main


if __name__ == "__main__":
    main()
