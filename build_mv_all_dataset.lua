-- takes the files in "th_AZ_VARIED", "th_EL_VARIED", "th_LIGHT_AZ_VARIED" and puts them,
-- renumbered, all into one folder.

-- not using this

require 'lfs'

DATASET_ROOT = "DATASET/TRANSFORMATION_DATASET"
OUTPUT_ROOT = DATASET_ROOT .. '/th_ALL'

os.execute('mkdir ' .. OUTPUT_ROOT)
os.execute('mkdir ' .. OUTPUT_ROOT .. '/FT_training')
os.execute('mkdir ' .. OUTPUT_ROOT .. '/FT_test')

specific_folder_names = {"th_AZ_VARIED", "th_EL_VARIED", "th_LIGHT_AZ_VARIED"}
TRAINING_NAME = "FT_training"
TESTING_NAME = "FT_test"


for _, mode_name in pairs({TRAINING_NAME, TESTING_NAME}) do
  current_index = 1
  for _, folder_name in pairs(specific_folder_names) do
    local current_directory = DATASET_ROOT .. '/' .. folder_name .. '/' .. mode_name
    for file_name in lfs.dir(current_directory) do
      local file_path = current_directory .. '/' .. file_name

      if lfs.attributes(file_path).mode == 'file' then
        local output_file_path = OUTPUT_ROOT .. '/' .. mode_name .. '/batch' .. current_index
        os.execute('cp ' .. file_path ..  ' ' .. output_file_path)

        current_index = current_index + 1
      end
    end
  end
end

