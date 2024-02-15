# apply black to pre-commit
pip install pre-commit
pre-commit sample-config > .pre-commit-config.yaml
cp pre-commit-config-template.yaml .pre-commit-config.yaml

# commit message template
git config --global commit.template .gitmessage.txt

# issue & pr template
mkdir .github
cp ISSUE_TEMPLATE.md .github/ISSUE_TEMPLATE.md
cp PULL_REQUEST_TEMPLATE.md .github/PULL_REQUEST_TEMPLATE.md

# remove files
rm ISSUE_TEMPLATE.md
rm PULL_REQUEST_TEMPLATE.md
rm pre-commit-config-template.yaml
rm gitsetting.tar