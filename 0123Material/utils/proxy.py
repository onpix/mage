# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://github.com/3DTopia/OpenLRM/blob/main/openlrm/utils/proxy.py

import os

NO_PROXY = "NO_DATA_PROXY" in os.environ

def no_proxy(func):
    NO_PROXY = True
    """Decorator to disable proxy but then restore after the function call."""
    def wrapper(*args, **kwargs):
        # http_proxy, https_proxy, HTTP_PROXY, HTTPS_PROXY, all_proxy
        http_proxy = os.environ.get('http_proxy')
        https_proxy = os.environ.get('https_proxy')
        HTTP_PROXY = os.environ.get('HTTP_PROXY')
        HTTPS_PROXY = os.environ.get('HTTPS_PROXY')
        all_proxy = os.environ.get('all_proxy')
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['all_proxy'] = ''
        try:
            return func(*args, **kwargs)
        finally:
            if http_proxy is not None:
                os.environ['http_proxy'] = http_proxy
            if https_proxy is not None:
                os.environ['https_proxy'] = https_proxy
            if HTTP_PROXY is not None:
                os.environ['HTTP_PROXY'] = HTTP_PROXY
            if HTTPS_PROXY is not None:
                os.environ['HTTPS_PROXY'] = HTTPS_PROXY
            if all_proxy is not None:
                os.environ['all_proxy'] = all_proxy
    if NO_PROXY:
        return wrapper
    else:
        return func