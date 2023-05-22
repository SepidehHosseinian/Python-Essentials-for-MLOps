# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

module_logger = logging.getLogger(__name__)


class Tree(object):
    """
    Helper in-mem tree built for traversal - if this class gets more complex,
    rethink the server-side queries. Should not be embedded into Runs until needed
    """
    class Node(object):
        def __init__(self, dto):
            self.dto = dto
            self.children = []

        def add_child(self, child):
            module_logger.debug("Setting %s as child of %s", child.dto.run_id, self.dto.run_id)
            self.children.append(child)

        def get_subtree(self, depth_id):
            module_logger.debug("Getting subtree for %s, which has %d children", self.dto.run_id, len(self.children))
            yield self, depth_id
            for child in self.children:
                module_logger.debug("Walking child %s", child.dto.run_id)
                for descendant, depth in child.get_subtree(depth_id + 1):
                    yield descendant, depth

        def _to_string_at_depth(self, depth):
            return "  " * depth + self.dto.run_id

        def __str__(self):
            tree = self.get_subtree(depth_id=0)
            return "\n".join(node._to_string_at_depth(depth) for node, depth in tree)

    def __init__(self, run_dtos):
        # Materialize if generator
        run_dtos = list(run_dtos)
        num_nodes = len(run_dtos)
        module_logger.debug("Forming tree with %d nodes", num_nodes)
        assert num_nodes > 0
        self._dtos = run_dtos

        # keep track of the root - guaranteed to exist
        root_id = run_dtos[0].root_run_id
        module_logger.debug("Constructing tree with root %s", root_id)

        # tree construction
        self._id_to_node = {dto.run_id: self.Node(dto) for dto in run_dtos}

        for rid, node in self._id_to_node.items():
            parent_id = node.dto.parent_run_id
            module_logger.debug("Run %s has parent %s", rid, parent_id)
            if parent_id is None:
                # If this condition isn't met, data model is corrupt
                assert rid == root_id
                self._root_node = node
            else:
                parent = self._id_to_node[parent_id]
                parent.add_child(node)

    def get_subtree(self, runid_or_dto=None):
        if runid_or_dto is None:
            rid = self._root_node.dto.run_id
            module_logger.debug("No subtree, returning full tree from %s", rid)
        elif hasattr(runid_or_dto, "run_id"):
            rid = runid_or_dto.run_id
            module_logger.debug("Getting subtree from Dto, returning tree from %s", rid)
        else:
            rid = runid_or_dto
            module_logger.debug("Assuming input is a run id: %s", runid_or_dto)

        subroot = self._id_to_node[rid]
        module_logger.debug("Getting subtree for %s", subroot)
        return subroot.get_subtree(depth_id=0)

    def get_subtree_dtos(self, runid_or_dto=None, skip_root=True):
        skip = skip_root
        for node, _ in self.get_subtree(runid_or_dto=runid_or_dto):
            if skip:
                skip = False
                continue
            yield node.dto

    def __str__(self):
        return str(self._root_node)
