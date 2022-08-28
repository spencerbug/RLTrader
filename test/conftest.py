import pytest
import git
import sys
import os

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

sys.path.insert(0, os.path.join(get_git_root("."),"src"))

# @pytest.fixture
# def exchange_server():