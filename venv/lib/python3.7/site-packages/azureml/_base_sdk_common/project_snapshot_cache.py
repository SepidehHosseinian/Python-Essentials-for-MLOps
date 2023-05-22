
import os
import json
import hashlib

import azureml._project.file_utilities as file_utilities

from azureml._base_sdk_common.merkle_tree import DirTreeNode
from azureml._base_sdk_common.utils import create_session_with_retry
from azureml.exceptions import ExperimentExecutionException


PROJECT_CONTENT_CACHE_DIR_NAME = ".projectcontent"
PROJECT_CONTENT_CACHE_LATEST_SNAPSHOT_FILE = "latestsnapshot"


class ContentSnapshotCache(object):
    def __init__(self, service_context):
        self.service_context = service_context

    def get_workspace_hash(self):
        h = hashlib.sha256()
        h.update(self.service_context.workspace_id.encode('utf-8'))
        return h.hexdigest()

    def _get_latest_snapshot_cache_file(self):
        return os.path.join(
            self.get_cache_directory(),
            PROJECT_CONTENT_CACHE_LATEST_SNAPSHOT_FILE)

    def get_cache_directory(self):
        return os.path.join(file_utilities.get_home_settings_directory(),
                            PROJECT_CONTENT_CACHE_DIR_NAME, self.get_workspace_hash())

    def get_latest_snapshot(self):
        latest_snapshot_cache_file = self._get_latest_snapshot_cache_file()
        if not os.path.isfile(os.path.abspath(latest_snapshot_cache_file)):
            # call the content service to fetch latest snapshot and populate the cache
            url = self.service_context._get_project_content_url() + \
                "/content/v1.0" + self.service_context._get_workspace_scope() + "/snapshots/latest/metadata"

            headers = self.service_context.get_auth().get_authentication_header()

            session = create_session_with_retry()
            response = session.get(url, headers=headers)
            if response.status_code >= 400:
                from azureml._base_sdk_common.common import get_http_exception_response_string
                raise ExperimentExecutionException(get_http_exception_response_string(response))

            response_data = response.content.decode('utf-8')
            snapshot_dict = json.loads(response_data)
            root_dict = snapshot_dict['root']
            snapshot_id = snapshot_dict['id']
            node = DirTreeNode()
            node.load_object_from_dict(root_dict)
        else:
            with open(latest_snapshot_cache_file, 'r') as json_data:
                d = json.load(json_data)
                root = d['root']
                snapshot_id = d['snapshot_id']
                node = DirTreeNode()
                node.load_root_object_from_json_string(root)
        return node, snapshot_id

    def update_cache(self, snapshot_dto):
        directory_path = self.get_cache_directory()
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            latest_snapshot_cache_file = self._get_latest_snapshot_cache_file()
            with open(latest_snapshot_cache_file, 'w+') as f:
                json.dump(snapshot_dto.__dict__, f)
        except FileExistsError:
            # concurrent snapshots competing for file, first can update cache
            pass

    def remove_latest(self):
        latest_snapshot_cache_file = self._get_latest_snapshot_cache_file()
        if os.path.exists(latest_snapshot_cache_file):
            os.remove(latest_snapshot_cache_file)
