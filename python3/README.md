# Cloud Shell Env
This creates a new cloudshell image.

## CommandLine 

### Test a cloudshell env

cloudshell env build-local

cloudshell env run

exit

### Push a cloudshell env

git commit -a -m ""

git push origin master

cloudshell env build-local

cloudshell env push

cloudshell env update-default-image
 