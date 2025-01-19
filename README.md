# UCI-ICS-GPU  
**Tutorial on using UCI SLURM Servers for GPU-accelerated tasks**  

## Requirements  
- ICS Openlab Account  
- Access to UCI SLURM servers  

---

## Step 1: Log into ICS Openlab Account  
1. Log in to your ICS Openlab account.  
2. Verify the correct user by running:  
   ```bash
   whoami
   ```  
   This should return your UCI Net ID.  

---

## Step 2: Submit a Job to UCI SLURM Servers with GPUs  
1. Check out the available SLURM servers here:  
   [UCI SLURM Wiki](https://wiki.ics.uci.edu/doku.php/services:slurm#running_a_shell_on_a_slurm_only_cluster)  
2. Load SLURM onto your Openlab instance by running the following commands:  
   ```bash
   module load slurm
   module initadd slurm
   ```  

---
## Step 3: Set Up the Environment  
1. Clone this repository or create your project. For this tutorial, we will use the Python file included in this repository.  
2. Load the Python environment:  
To submit the enviorment load script, use the following command:  
```bash
sbatch -p opengpu.p load.sh
``` 
Running this on the opengpu server will ensure that we correctly load all binary
Output of this command will be located in a file called "slurm-{jobnumber}.out" where the job number is of the job you just loaded
---

## Step 4: Running the SLURM Job  
We will use an SBATCH script to submit the job. Keep in mind:  
- SBATCH "comments" (lines starting with `#SBATCH`) are not just comments; they configure the nodes assigned by the SLURM instance.  
- Detailed documentation can be found [here](https://wiki.ics.uci.edu/doku.php/services:slurm#single_core_single_gpu_job).  

To submit the job, use the following command:  
```bash
sbatch -p opengpu.p slurmscript.sh
```  
Replace `slurmscript.sh` with the name of your SLURM script or Python file.  

---

## Step 5: Viewing Output and Errors  
To check the output and errors generated by the SLURM script:  
- View output logs:  
  ```bash
  tail -n +1 -f myoutput.out
  ```  
- View error logs:  
  ```bash
  tail -n +1 -f myerrors.err
  ```  

---

### Additional Resources  
- [UCI SLURM Wiki](https://wiki.ics.uci.edu/doku.php/services:slurm#single_core_single_gpu_job)  

