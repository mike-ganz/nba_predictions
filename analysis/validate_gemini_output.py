import os
import glob
import json
from typing import List, Tuple


def find_latest_ultra_fast_file() -> str:
    pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training', 'nba_2023_2024_gemini_compact_remaining_plays_ULTRA_FAST_*.jsonl')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError('No ULTRA_FAST gemini output file found under data/training')
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def validate_user_context(user_json: dict) -> Tuple[bool, List[str]]:
    ok = True
    errs: List[str] = []
    # Required keys
    for key in ['A', 'H', 'as', 'hs', 'ap', 'hp', 'L', 'p', 'tb', 'sd', 'pos']:
        if key not in user_json:
            ok = False
            errs.append(f"missing key: {key}")

    # Types and values
    pos = user_json.get('pos')
    if pos not in ('A', 'H', 'N'):
        ok = False
        errs.append(f"pos invalid: {pos}")

    tb = user_json.get('tb')
    if not (isinstance(tb, list) and len(tb) == 2 and all(isinstance(x, int) for x in tb)):
        ok = False
        errs.append(f"tb invalid: {tb}")

    sd = user_json.get('sd')
    if not isinstance(sd, int):
        ok = False
        errs.append(f"sd invalid: {sd}")

    p = user_json.get('p', [])
    # 7-element tuples check and o_foul pairing
    for idx, t in enumerate(p):
        if not (isinstance(t, list) and len(t) == 7):
            ok = False
            errs.append(f"tuple length !=7 at p[{idx}]: {t}")
        if isinstance(t, list) and len(t) >= 5 and t[4] == 'o_foul':
            # Next must be tov with same q,t,score,actor, lineup
            if idx + 1 >= len(p):
                ok = False
                errs.append(f"o_foul at p[{idx}] not followed by tov (end of list)")
            else:
                t2 = p[idx + 1]
                if not (isinstance(t2, list) and len(t2) >= 5 and t2[4] == 'tov' and t2[0] == t[0] and t2[1] == t[1] and t2[2] == t[2] and t2[3] == t[3] and t2[6] == t[6]):
                    ok = False
                    errs.append(f"o_foul at p[{idx}] not immediately paired with tov: {t2}")

    return ok, errs


def validate_model_output(model_text: str) -> Tuple[bool, List[str]]:
    ok = True
    errs: List[str] = []
    try:
        arr = json.loads(model_text)
        if not (isinstance(arr, list) and len(arr) == 7):
            ok = False
            errs.append(f"model tuple not 7 elements: {arr}")
    except json.JSONDecodeError as e:
        ok = False
        errs.append(f"model text not JSON: {e}")
    return ok, errs


def main() -> None:
    path = find_latest_ultra_fast_file()
    print(f"Validating file: {path}")
    user_ok_total = 0
    model_ok_total = 0
    user_errs_total: List[str] = []
    model_errs_total: List[str] = []

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                contents = obj.get('contents', [])
                user_text = contents[0]['parts'][0]['text']
                model_text = contents[1]['parts'][0]['text']
                user_json = json.loads(user_text)

                u_ok, u_errs = validate_user_context(user_json)
                m_ok, m_errs = validate_model_output(model_text)

                if u_ok:
                    user_ok_total += 1
                else:
                    user_errs_total.extend([f"line {i}: {e}" for e in u_errs])

                if m_ok:
                    model_ok_total += 1
                else:
                    model_errs_total.extend([f"line {i}: {e}" for e in m_errs])

            except Exception as e:
                user_errs_total.append(f"line {i}: exception {e}")

            if i >= 2000:
                break

    print(f"User contexts valid: {user_ok_total}")
    print(f"Model outputs valid: {model_ok_total}")
    if user_errs_total:
        print("User errors:")
        for e in user_errs_total[:20]:
            print(" - ", e)
    if model_errs_total:
        print("Model errors:")
        for e in model_errs_total[:20]:
            print(" - ", e)


if __name__ == '__main__':
    main()


