from pytorch_image_generation_metrics import (
    get_inception_score_from_directory,
    get_fid_from_directory,
    get_inception_score_and_fid_from_directory)

ISa, IS_stda = get_inception_score_from_directory(
    'data/metaverse/original')

ISb, IS_stdb = get_inception_score_from_directory(
    'data/metaverse/generated')

print(f'Inception Score Original: {ISa} ± {IS_stda}')
print(f'Inception Score Generated: {ISb} ± {IS_stdb}')