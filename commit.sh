#simple shell script to auto commit

git remote set-url origin git@github.com:murchie85/murchie85.github.io.git
git remote add origin https://murchie85:commando85@github.com/murchie85/murchie85.github.io.git

git add .

echo 'Enter the commit message:'
#dollar1 becomes the commit message

git commit -m "$1"

#echo 'Enter the name of the branch:'
#read branch

git push origin master

