import os
from urllib.parse import unquote


def find_label_dirs(label_root):
    """Find leaf directories that contain good/bad label files."""
    label_dirs = []
    for root, _dirs, files in os.walk(label_root):
        has_label = any(name.startswith(("good-", "bad-")) and name.endswith(".txt") for name in files)
        if has_label:
            label_dirs.append(root)
    return sorted(label_dirs)


def parse_label_file(file_path):
    """Parse label file and return records with pixel coords and filenames."""
    records = []
    aligned_filename = None
    template_aligned_filename = None
    file_dir = None
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            file_dir = parts[1]
            aligned_filename = parts[2]
            template_aligned_filename = parts[3]
            pixel_x = float(parts[8])
            pixel_y = float(parts[9])
            records.append((pixel_x, pixel_y))
    return file_dir, aligned_filename, template_aligned_filename, records


def resolve_data_path(file_dir, filename, data_root_override):
    """Resolve data path with optional root override."""
    dir_candidates = []
    name_candidates = []
    if file_dir:
        dir_candidates.extend([file_dir, unquote(file_dir)])
    if filename is not None and filename != "":
        name_candidates.extend([filename, unquote(filename)])
    if filename == "":
        for directory in dir_candidates:
            if directory and os.path.isdir(directory):
                return directory

    for directory in dir_candidates:
        if directory and os.path.isdir(directory):
            for name in name_candidates:
                candidate = os.path.join(directory, name)
                if os.path.exists(candidate):
                    return candidate

    if data_root_override and os.path.isdir(data_root_override):
        drive, tail = os.path.splitdrive(file_dir or "")
        if drive and tail:
            rel = tail.lstrip("\\/")
            for name in name_candidates:
                candidate = os.path.join(data_root_override, rel, name)
                if os.path.exists(candidate):
                    return candidate

        base_dir = os.path.basename(file_dir or "")
        if base_dir:
            for name in name_candidates:
                candidate = os.path.join(data_root_override, base_dir, name)
                if os.path.exists(candidate):
                    return candidate

    return None
