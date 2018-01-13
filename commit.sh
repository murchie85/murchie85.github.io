#simple shell script to auto commit


git add .

echo 'Enter the commit message:'
#dollar1 becomes the commit message

git commit -m "$1"

#echo 'Enter the name of the branch:'
#read branch

git push origin master

