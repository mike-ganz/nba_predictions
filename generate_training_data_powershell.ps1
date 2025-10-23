# ./generate_training_data_powershell.ps1
python generate_2023_2024_season.py --n-total 15 --format gemini --generation-mode remaining_plays --season 2023-2024 --ultra-fast-gemini
python generate_2023_2024_season.py --n-total 15 --format gemini --generation-mode first_N_plays --season 2023-2024 --ultra-fast-gemini
python generate_2023_2024_season.py --n-total 15 --format gemini --generation-mode first_N_plays --season 2022-2023 --ultra-fast-gemini
python generate_2023_2024_season.py --n-total 15 --format gemini --generation-mode first_N_plays --season 2021-2022 --ultra-fast-gemini