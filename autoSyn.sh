#!/bin/sh
git status  
git add *  
git commit -m 'add some code from PC'
# git commit -m 'add some results from Server'
git pull --rebase origin master   #domnload data
git push origin master            #upload data
git stash pop