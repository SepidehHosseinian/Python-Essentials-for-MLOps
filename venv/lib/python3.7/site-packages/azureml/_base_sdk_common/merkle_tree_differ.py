# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from os.path import join


class DiffResultEntry(object):
    def __init__(self, operation_type, node_path, is_file=True):
        self.operation_type = operation_type
        self.node_path = node_path
        self.is_file = is_file

    def __eq__(self, other):
        """Override the default Equals behavior"""
        return (self.operation_type == other.operation_type
                and self.node_path == other.node_path and self.is_file == other.is_file)


def compute_diff(prev, curr):
    entries = []
    compute_diff_helper(prev, curr, entries, '')
    return entries


''' Diff between files at this level. This assumes that the files are sorted by name in the given list'''


def diff_files_at_level(prev_files, curr_files, entries, prefix):
    curr_pointer = 0
    prev_pointer = 0
    while (curr_pointer < len(curr_files) and prev_pointer < len(prev_files)):
        if (curr_files[curr_pointer].name == prev_files[prev_pointer].name):
            if (prev_files[prev_pointer].hexdigest_hash != curr_files[curr_pointer].hexdigest_hash):
                entries.append(DiffResultEntry('modified', join(prefix, curr_files[curr_pointer].name)))
            prev_pointer = prev_pointer + 1
            curr_pointer = curr_pointer + 1
        elif (curr_files[curr_pointer].name > prev_files[prev_pointer].name):
            entries.append(DiffResultEntry('removed', join(prefix, prev_files[prev_pointer].name)))
            prev_pointer = prev_pointer + 1
        elif (curr_files[curr_pointer].name < prev_files[prev_pointer].name):
            entries.append(DiffResultEntry('added', join(prefix, curr_files[curr_pointer].name)))
            curr_pointer = curr_pointer + 1
    while (curr_pointer < len(curr_files)):
        entries.append(DiffResultEntry('added', join(prefix, curr_files[curr_pointer].name)))
        curr_pointer = curr_pointer + 1
    while (prev_pointer < len(prev_files)):
        entries.append(DiffResultEntry('removed', join(prefix, prev_files[prev_pointer].name)))
        prev_pointer = prev_pointer + 1


''' Diff between dirs at this level. This assumes that the dirs are sorted by name in the given list'''


def diff_dirs_at_level(curr_dirs_at_level, prev_dirs_at_level, entries, prefix):
    curr_pointer = 0
    prev_pointer = 0
    while (curr_pointer < len(curr_dirs_at_level) and prev_pointer < len(prev_dirs_at_level)):
        if (curr_dirs_at_level[curr_pointer].name == prev_dirs_at_level[prev_pointer].name):
            if (prev_dirs_at_level[prev_pointer].hexdigest_hash != curr_dirs_at_level[curr_pointer].hexdigest_hash):
                compute_diff_helper(
                    prev_dirs_at_level[prev_pointer],
                    curr_dirs_at_level[curr_pointer],
                    entries,
                    join(prefix, prev_dirs_at_level[prev_pointer].name))
            prev_pointer = prev_pointer + 1
            curr_pointer = curr_pointer + 1
        elif (curr_dirs_at_level[curr_pointer].name > prev_dirs_at_level[prev_pointer].name):
            populate_add_or_remove_entries_for_all_files_in_dir(
                entries,
                curr_dirs_at_level[curr_pointer],
                join(prefix, curr_dirs_at_level[curr_pointer].name),
                "added")
            curr_pointer = curr_pointer + 1
        elif (curr_dirs_at_level[curr_pointer].name < prev_dirs_at_level[prev_pointer].name):
            populate_add_or_remove_entries_for_all_files_in_dir(
                entries,
                prev_dirs_at_level[prev_pointer],
                join(prefix, prev_dirs_at_level[prev_pointer].name),
                "removed")
            prev_pointer = prev_pointer + 1
    while (curr_pointer < len(curr_dirs_at_level)):
        populate_add_or_remove_entries_for_all_files_in_dir(
            entries,
            curr_dirs_at_level[curr_pointer],
            join(prefix, curr_dirs_at_level[curr_pointer].name),
            "added")
        curr_pointer = curr_pointer + 1
    while (prev_pointer < len(prev_dirs_at_level)):
        populate_add_or_remove_entries_for_all_files_in_dir(
            entries,
            prev_dirs_at_level[prev_pointer],
            join(prefix, prev_dirs_at_level[prev_pointer].name),
            "removed")
        prev_pointer = prev_pointer + 1


def compute_diff_helper(prev, curr, entries, prefix):
    if (prev.hexdigest_hash == curr.hexdigest_hash):
        return
    prev_files_at_level = [t for t in prev.children if t.file_type == "File"]
    curr_files_at_level = [t for t in curr.children if t.file_type == "File"]

    diff_files_at_level(prev_files_at_level, curr_files_at_level, entries, prefix)

    prev_dirs_at_level = [t for t in prev.children if t.file_type == "Directory"]
    curr_dirs_at_level = [t for t in curr.children if t.file_type == "Directory"]

    diff_dirs_at_level(curr_dirs_at_level, prev_dirs_at_level,  entries, prefix)


def populate_add_or_remove_entries_for_all_files_in_dir(entries, root, prefix, operation_type):
    entries.append(DiffResultEntry(operation_type, prefix, is_file=False))
    for child in root.children:
        node_path = join(prefix, child.name)
        if (child.is_file()):
            entries.append(DiffResultEntry(operation_type, node_path))
        else:
            populate_add_or_remove_entries_for_all_files_in_dir(entries, child, node_path, operation_type)
