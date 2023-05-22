# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from collections import OrderedDict
import json


class ArmTemplateBuilder(object):

    def __init__(self):
        template = OrderedDict()
        template['$schema'] = \
            'https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#'
        template['contentVersion'] = '1.0.0.0'
        template['parameters'] = {}
        template['variables'] = {}
        template['resources'] = []
        self.template = template

    def add_resource(self, resource):
        self.template['resources'].append(resource)

    def build(self):
        return json.loads(json.dumps(self.template, default=set_default))


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def build_storage_account_resource(name, location):
    storage_account = {
        'type': 'Microsoft.Storage/storageAccounts',
        'name': name,
        'apiVersion': '2016-12-01',
        'location': location,
        'sku': {'name': 'Standard_LRS'},
        'kind': 'Storage',
        'dependsOn': [],
        'properties': {
            "encryption": {
                "services": {
                    "blob": {
                        "enabled": 'true'
                    }
                },
                "keySource": "Microsoft.Storage"
            },
            "supportsHttpsTrafficOnly": True,
            "allowBlobPublicAccess": False
        }
    }
    return storage_account


def build_team_resource(name, location, storageAccountId, seatCount, storageName):
    team_resource = {
        'type': 'Microsoft.MachineLearningServices/accounts',
        'name': name,
        'apiVersion': '2018-03-01-preview',
        'location': location,
        'resources': [],
        'dependsOn': [],
        'properties': {
                'seats': seatCount,
                'storageAccount': {
                    'storageAccountId': storageAccountId,
                    'accessKey': "[listKeys(resourceId('{}/{}', '{}'), '2016-12-01').keys[0].value]".format(
                        'Microsoft.Storage', 'storageAccounts', storageName)
                }
        }
    }
    return team_resource
