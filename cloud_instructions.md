---
output:
  pdf_document: default
  html_document: default
---
# Logging into the MLDSP cloud
## 1. Make a compute canada account if you dont already have one 
Compute canada account: https://ccdb.computecanada.ca/security/login please use your institutional email, under **Position** select only from the **Sponsored Users** options your appropriate status and submit the application.

Dr. Hill or Dr. Kari will need approve your account via email.
## 2. If you need access to the OpenStack cloud dashboard 
### 2a.
In the compute canada account https://ccdb.computecanada.ca homepage (after login) select the 'My Account' menu and from the dropdown options select 'Manage RAP membership'. On this page promote yourself to manager for the RAP **dbh-824-ab** (from dropdown menu). 

If you do not have permission, message someone with a manager role to do this for you.

### 2b. 
Read this page on using the cloud https://docs.computecanada.ca/wiki/Cloud#Using_the_Cloud, for reference our cloud is on the graham cloud. Fill out the form mentioned under "Getting a cloud project": https://docs.google.com/forms/d/e/1FAIpQLSeU_BoRk5cEz3AvVLf3e9yZJq-OvcFCQ-mg7p4AWXmUkd5rTw/viewform **IMPORTANT:** in the google form fill out the title as such: **dbh-824**

Compute canada will either contact you or your PI to approve your account for cloud access and allow your IP address to access the cloud dashboard: https://graham.cloud.computecanada.ca. 

If you do not see a login page your IP (found at: http://ipv4.icanhazip.com/) isn't whitelisted (you can only access cloud dashboards from the IP you provide which may change if your router/Internet is reset or you're logging in from a different location than the one from which you provided your IP). Please send a support ticket to compute canada to whitelist your IP for the graham cloud 

## 3. Accessing the Virtual Machines/VM with ssh 
### subject to change as our Network structures changes
Currently there is 1 VM running (ubuntu 18.04) with a root user 'ubuntu' and several linux user accounts for lab members. 

If you need a user account, someone with VM access(manager) will need to make you an account (instructions below), otherwise you can use the root user to login. 

In either case you need to make a set of ssh keys instructions:https://docs.computecanada.ca/wiki/SSH_Keys/en follow up to 'Generating an SSH key' step then send the **PUBLIC key only** to a manager so they may upload it to your account/root account. **If you accidentally send the private key out EVEN ONCE, please delete the key pair and make a fresh set.**

Send your public IP found here: http://ipv4.icanhazip.com, for a manager to add your IP to the ubuntu account you need to access.

Login: ```ssh -i /path/where/your/private/key/is/my_key <user name>@<public IP of your server>``` user name will be either 'ubuntu' for root or your provided user and the server public IP is 199.241.167.171

If you get an error 'timout port 22' or similar either your IP or ssh keys do not match, verify with a manager


## Extra stuff
### Instructions for making new user on ubuntu
Use ```sudo adduser username``` to make ubuntu user account.  ```ssh-copy-id``` doesnt work for cloud servers due to compute canada security rules

Requesting users to make a pair of ssh keys on their machine (see CC wiki) and send you only the public key.

Request users give you their IP address by using: http://ipv4.icanhazip.com, add their address to the security group rules on open stack

To set up user's account run these commands replacing 'vivek' with their username. The ubuntu password set for all users is 'khill22', they shouldn't have to use this. Their login password will be whatever they set for their ssh keys
```
sudo mkdir /home/vivek/.ssh/
sudo chmod 0700 /home/vivek/.ssh/
sudo sh -c "echo 'copy user's PUBLIC key' > /home/vivek/.ssh/authorized_keys"
sudo chown -R vivek:vivek /home/vivek/.ssh/ 
```
### Instructions for spinning up a VM
https://docs.computecanada.ca/wiki/Cloud_Quick_Start 