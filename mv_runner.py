import os

jobs = [
        {
            'shape_bias': True,
            'shape_bias_amount': 10,
            'import': 'picasso'
        }
    ]

for job in jobs:
    jobname = "MV"
    flagstring = ""
    for flag in job:
        if isinstance(job[flag], bool):
            if job[flag]:
                jobname = jobname + "_" + flag
                flagstring = flagstring + " --" + flag
            else:
                print "WARNING: Excluding 'False' flag " + flag
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
    flagstring = flagstring + " --save " + jobname

    print ("th monovariant_main.lua" + flagstring)
    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("th monovariant_main.lua" + flagstring)

    # if not os.path.exists(jobname):
    #     os.makedirs(jobname)

    # with open(jobname + '/generating_parameters.txt', 'w') as paramfile:
    #     paramfile.write(str(job))

    if True:
        os.system("sbatch -N 1 -c 2 --gres=gpu:1 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
