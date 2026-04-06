import os

def type_switch(base_dir="../data/test", old_prefix="group_", new_prefix="real_", dry_run=False):
    """Batch rename entries in base_dir by replacing old_prefix with new_prefix."""
    base_dir = os.path.abspath(base_dir)

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    renamed = 0
    skipped = 0

    for name in os.listdir(base_dir):
        src = os.path.join(base_dir, name)

        # Only rename directories here; remove this check if you also want files.
        if not os.path.isdir(src):
            skipped += 1
            continue

        if name.startswith(old_prefix):
            new_name = new_prefix + name[len(old_prefix):]
        elif name.startswith(new_prefix):
            # Already converted.
            skipped += 1
            continue
        else:
            # Keep original suffix and add new prefix.
            new_name = new_prefix + name

        dst = os.path.join(base_dir, new_name)

        if os.path.exists(dst):
            print(f"[SKIP] target exists: {dst}")
            skipped += 1
            continue

        if dry_run:
            print(f"[DRY-RUN] {src} -> {dst}")
        else:
            os.rename(src, dst)
            print(f"[OK] {src} -> {dst}")
        renamed += 1

    print(f"Done. renamed={renamed}, skipped={skipped}, dir={base_dir}")


if __name__ == "__main__":
    # Preview only (no changes): type_switch(dry_run=True)
    type_switch()