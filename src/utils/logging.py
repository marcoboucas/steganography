class Color:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Symbol:
    SUCCESS = f"{Color.OKGREEN}✓{Color.ENDC}"
    FAIL = f"{Color.FAIL}✗{Color.ENDC}"
    SKIP = f"{Color.WARNING}⚠{Color.ENDC}"
