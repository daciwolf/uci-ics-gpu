# uci-ics-gpu
Tutorial on using UCI Slurm Servers for GPU accelerated tasks



# Requirements, ICS Openlab Account 

# Step 1. 

Log into ICS Openlab Account 

verify correct user with: $ whoami

You should see your uci net id

# Step 2. 

Create your project, for this tutorial we will be using a the python file located in this repo

For our project we will need to load a python enviornment 

Run the following command to change the load script to an executable: "chmod + x load.sh"

Then run the load script: "./load.sh"

# Step 3. 

Now we need to begin submitting a job to the uci SLURM servers with GPUS 

you can check out the available servers at:    https://wiki.ics.uci.edu/doku.php/services:slurm#running_a_shell_on_a_slurm_only_cluster



first we need to load slurm onto our instance openlab however: 

Run the following 2 commands: 
1. " module load slurm "
2. " module initadd slurm "


