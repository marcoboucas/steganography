"""Text utils functions."""


def charac_to_bin(character: str) -> str:
    """To binary."""
    return bin(ord(character))[2:].zfill(8)


def bin_to_charac(binary: str) -> str:
    """To character."""
    return chr(int(binary, 2))


def convert_str_to_octet(txt: str) -> str:
    """Convert str to octets."""
    return "".join(list(map(charac_to_bin, txt)))


def convert_octet_to_str(txt: str) -> str:
    """Conver octets to str."""
    return "".join([bin_to_charac(txt[i : i + 8]) for i in range(0, len(txt), 8)])
