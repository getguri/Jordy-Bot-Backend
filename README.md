Push an existing Git repository

cd existing_repo
git remote rename origin old-origin
git remote add origin git@gitlab.hcl-edtech.com:eboxdev/genai-sandbox-backend.git
git push -u origin --all
git push -u origin --tags

Create a new repository

git clone git@gitlab.hcl-edtech.com:eboxdev/genai-sandbox-backend.git
cd genai-sandbox-backend
git switch -c main
touch README.md
git add README.md
git commit -m "add README"
git push -u origin main
