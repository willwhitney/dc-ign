import os

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")


networks_dir = '/om/user/wwhitney/facegen_networks/'
base_networks = {
        'picasso': networks_dir + 'picasso',
        'braque': networks_dir + 'braque'
    }

# Don't give it a `save` name - that gets generated for you
jobs = [
        {
            'import': 'picasso',
	    'clearQ': True,
        },
        {
            'import': 'braque',
            'dim_hidden': 120,
            'clearQ': True,
        },
        {
            'import': 'picasso',
            'learning_rate': -0.001,
            'clearQ': True,
        },
        {
            'import': 'braque',
            'dim_hidden': 120,
            'learning_rate': -0.001,
            'clearQ': True,
        },
        {
            'no_load': True,
            'clearQ': True,
        },
        {
            'no_load': True,
            'learning_rate': -0.001,
            'clearQ': True,
        },
        {
            'no_load': True,
            'shape_bias': True,
            'shape_bias_amount': 10,
            'clearQ': True,
        },
        {
            'no_load': True,
            'shape_bias': True,
            'shape_bias_amount': 10,
            'learning_rate': -0.001,
            'clearQ': True,
        },
        {
            'no_load': True,
            'shape_bias': True,
            'shape_bias_amount': 80,
            'clearQ': True,
        },
        {
            'no_load': True,
            'shape_bias': True,
            'shape_bias_amount': 80,
            'learning_rate': -0.001,
            'clearQ': True,
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
        elif flag == 'import':
            imported_network_name = job[flag]
            if imported_network_name in base_networks.keys():
                network_location = base_networks[imported_network_name]
                jobname = jobname + "_" + flag + "_" + str(imported_network_name)
                flagstring = flagstring + " --" + flag + " " + str(network_location)
            else:
                jobname = jobname + "_" + flag + "_" + str(job[flag])
                flagstring = flagstring + " --" + flag + " " + str(job[flag])
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
    flagstring = flagstring + " --save " + jobname


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

    print ("th monovariant_main.lua" + flagstring)
    if True:
        os.system("sbatch -N 1 -c 2 --gres=gpu:1 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")




