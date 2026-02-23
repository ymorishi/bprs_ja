# load modules
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
import pandas as pd
import yaml, json, os
import numpy as np


def main() -> None:
    # import settings
    currdir=os.getcwd()
    yml_path=os.path.join(currdir,"bprs_config.yaml")
    print('yaml path {pth}'.format(pth=yml_path))
    with open(yml_path, encoding='utf-8')as f:
        config = yaml.safe_load(f)
    print('config file loaded')

    # %%
    # Specify data directry
    data_dir = config["data"]["dataset_dir"]
    model_dir = config["model"]["model_dir"]

    # %%
    model_id=config["model"]["model_name"]
    model_path = os.path.join(model_dir,model_id.split("/")[-1])

    # some cumtom variables
    quant = config["model"]["quantization"]
    kv_quant=config["inference"]["kv_quant"],
    batch_size = config["inference"]["batch_size"],
    llm_kwargs={
            "tensor_parallel_size": config["model"]["tensor_parallel_size"], 
            "max_model_len": config["inference"]["max_model_len"], # 65536,32768 for stockmark 
            "enforce_eager":True, # vLLM doesn’t invoke torch.compile Throughput is a bit lower but the worker stays responsive.
            "max_num_seqs": config["inference"]["batch_size"],
            "gpu_memory_utilization": config["model"]["gpu_memory_utilization"],
    }  
    # add quantization option if exists

    if quant:
        llm_kwargs["quantization"] = quant

    if kv_quant:
        llm_kwargs["kv_cache_dtype"]=kv_quant
    
    if llm_kwargs["tensor_parallel_size"]>1:
        llm_kwargs["distributed_executor_backend"]="mp"
        llm_kwargs["disable_custom_all_reduce"] = True

    print(llm_kwargs)

    # %%
    # load model with vLLM

    #from huggingface_hub import login
    #HF_TOKEN="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"
    #login(HF_TOKEN)
    #    os.environ["VLLM_USE_FLASHINFER"] = "0"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    print('tokenizer loaded')

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        download_dir=model_path,
        **llm_kwargs
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        top_k=40,
        max_tokens=config["inference"]["max_tokens"],
    )

    # %%
    # Import data
    ds=load_dataset('json', data_files= os.path.join(data_dir,config["data"]["dataset_name"]))

    # %%
    system_prompt = """
    あなたは日本語での精神科臨床に熟達した評価者です。
    対象は日本の精神科入院EHRに記載された「患者の発話に相当する文」です。
    目的は各文について、Brief Psychiatric Rating Scale（BPRS）日本語版の18項目をもとに、症状の該当性・強度を0〜1で評定することです。
    必ず以下の評価項目と定義・評定ルール・出力仕様を厳守してください。
    特に評価項目の定義での「評価対象」、「評価しないもの」に特に注意して評価してください。

    【評価項目と定義】
    1. somatic_concern（心気症）
    定義：身体的健康に対する過度な心配や訴え。
    評価対象：身体的な訴えや心配の発話内容。実際の身体疾患の有無は問わない。
    評価しないもの：気分の落ち込みや不安に伴う一般的な心配。

    2. anxiety（不安）
    定義：緊張・恐怖・心配などの主観的体験。
    評価対象：発話で示される不安・恐怖・焦燥感。
    評価しないもの：単なる落ち着かなさや身体動作の観察所見（tensionで評価）。
    発話に「怖い」「不安」「落ち着かない」「心配だ」などが頻出する場合に中等度以上を検討。

    3. emotional_withdrawal（情動的引きこもり）
    定義：他者との情緒的関係の欠如、関心や温かみの低下。
    評価対象：発話や表現における感情的距離・他者への無関心。
    評価しないもの：単に話が短い、主題が限られるだけの会話。
    他者との交流拒否や感情的遮断の明確な表現がある場合のみ上げる。
    「対人共感性、会話の開放性が欠如しており、面接者への親近感、関心、関与がきわめて少ない。対人距離ならびに言語的および非言語的コミュニケーションの減少によって明らかになる
    
    4. conceptual_disorganization（概念の統合障害）
    定義：発話のまとまりのなさ、論理的なつながりの欠如。
    評価対象：文脈から逸脱した発言、支離滅裂な思考の痕跡。
    評価しないもの：単なる省略・曖昧な表現・言葉足らず。
    発話の中で話題の飛躍や論理不整合が頻繁にみられる場合に0.4以上。

    5. guilt_feelings（罪責感）
    定義：過去の行為に対する過度な罪悪感や自己非難。
    評価対象：発話で「自分のせい」「迷惑をかけた」などを繰り返す内容。
    評価しないもの：一般的な反省や軽い後悔。
    内容が非現実的・誇大的であればunusual_thought_contentも同時に上げる。

    6. tension（緊張）
    定義：観察される身体的・運動的な緊張や落ち着かなさ。
    評価対象：身体的落ち着かなさ・身振りなどの観察情報。
    発話のみでは評価しない。
    「そわそわしている」「手をもじもじしている」など明示される描写がある場合のみ評価。

    7. mannerisms_and_posturing（常同動作および姿勢）
    定義：奇妙・不自然な姿勢や反復動作。
    評価対象：観察された身体動作。
    発話のみでは評価しない。

    8. grandiosity（誇大性）
    定義：過度な自己評価や特別な能力・地位への信念。
    評価対象：発話での自己誇示や特別な使命感の訴え。
    評価しないもの：単なる自信や希望的発言。
    「自分は選ばれた」「特別な力がある」など非現実的内容で0.4以上。

    9. depressive_mood（抑うつ気分）
    定義：悲しみ、興味喪失、絶望感、無価値感。
    評価対象：発話内容。
    評価しないもの：単なる疲労感や不満。
    「悲しい」「死にたい」「何も楽しくない」などが頻繁に現れる場合に0.4以上。

    10. hostility（敵意）
    定義：他者に対する怒り、攻撃的態度、敵対的発言。
    評価対象：発話における攻撃的語調や敵意の内容。
    評価しないもの：正当な不満表明。
    「むかつく」「殴ってやりたい」「誰も信用できない」など頻繁に出る場合に上げる。

    11. suspiciousness（猜疑心）
    定義：他人に対する被害的または不信的な考え。
    評価対象：発話内容。
    評価しないもの：一般的な慎重さや不安。
    「監視されている」「誰かが自分を傷つけようとしている」などの具体的表現がある場合に0.4以上。

    12. hallucinatory_behavior（幻覚による行動）
    定義：幻聴・幻視などに基づく行動・言動。
    評価対象：発話および行動描写。
    発話のみで「声が聞こえる」「見えないものが見える」と語る場合は中等度（0.4〜0.6）。
    実際の反応行動（呼びかけに返答する、空間に話しかける）が記述されている場合に高得点。

    13. motor_retardation（運動減退）
    定義：動作の遅延や身体的活動性の低下。
    評価対象：観察された行動のみ。
    「疲れた」「だるい」などの主観的表現は含めない。

    14. uncooperativeness（非協調性）
    定義：面接や対話への抵抗・協力の欠如。
    評価対象：発話内容（拒否的・攻撃的発言）および行動描写。
    「話したくない」「もういい」「聞きたくない」などが繰り返される場合に評価。

    15. unusual_thought_content（不自然な思考内容）
    定義：非現実的・奇異・妄想的な思考内容。
    評価対象：発話内容。
    「電波で操られている」「特別な使命がある」「他人の心が読める」など。
    強い確信・生活への影響があるほど高得点。

    16. blunted_affect（情動の平板化）
    定義：感情表現の乏しさ、声や表情の単調さ。
    評価対象：観察された表情・声調。
    描写に「表情がない」「声に抑揚がない」などが含まれる場合に評価。

    17. excitement（興奮）
    定義：感情の高ぶり、落ち着きのなさ、話しすぎ。
    評価対象：観察された発話量・態度。
    「大声」「早口」「止まらない」「ハイテンション」と記述される場合に上げる。

    18. disorientation（失見当識）
    定義：人・場所・時間の誤認や混乱。
    評価対象：発話内容で誤った時制・状況判断が明示される場合。
    明確な混乱（「ここは家ですか？」など）がある場合に上げる。

    【評定方法】
    - 各項目を**0.0〜1.0（少数1桁）**で評定。
        0.0：該当なし
        0.1〜0.3：軽度（わずかな該当）
        0.4〜0.6：中等度（明確に認められる）
        0.7〜0.9：重度（頻繁または顕著）
        1.0：最重度（持続的・生活機能を著しく損なう）
    - 評価は該当性と強度の総合として判断する。
    - 該当情報が不明確な場合は過剰推測を避け、0.0〜0.2にとどめる。
    - 観察された行動のみで評価する項目は、発話内容のみでは0.0とする。

    【言語上の注意】
    - 省略や口語（「だるい」「まあまあ」「別に」）は文脈的に解釈する。
    -「特にない」「変わりなし」など定型否定は、blunted_affectやemotional_withdrawalを上げる可能がある。
    -「眠れない」「落ち着かない」「怖い」→ anxiety、「疲れた」「だるい」→ somatic_concernに寄与。
    -「声が聞こえる」「人に見られている」→ hallucinatory_behavior, suspiciousness。
    - 妄想的内容（例：「自分は神だ」）→ grandiosity または unusual_thought_content。

    【出力仕様（厳守）】
    - すべての項目を**数値（少数1桁）**で返す。欠損禁止。
    - 出力は次のJSON形式のみ。余分な文字・説明は禁止。
    - フィールド順は固定。
        {
        "somatic_concern": 0.0,
        "anxiety": 0.0,
        "emotional_withdrawal": 0.0,
        "conceptual_disorganization": 0.0,
        "guilt_feelings": 0.0,
        "tension": 0.0,
        "mannerisms_posturing": 0.0,
        "grandiosity": 0.0,
        "depressive_mood": 0.0,
        "hostility": 0.0,
        "suspiciousness": 0.0,
        "hallucinatory_behavior": 0.0,
        "motor_retardation": 0.0,
        "uncooperativeness": 0.0,
        "unusual_thought_content": 0.0,
        "blunted_affect": 0.0,
        "excitement": 0.0,
        "disorientation": 0.0
        }
 
     """

    # prompt functions
    def user_prompt_v1(sentence):
        user_prompt = f"""以下の患者発話文を評価してください。JSONのみで出力してください。
        文: 「{sentence}」
        """
        return user_prompt

    def format_chat(system_prompt,sentence):
        user_prompt = user_prompt_v1(sentence)
        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def dataset_to_prompt(dataset,system_prompt):

        sentences_ja: list[str] = list(dataset["train"]["PtSpeech"])
        # apply chat template
        prompts = [format_chat(system_prompt,sentence) for sentence in sentences_ja]
        print('applying chat template: completed')
        return prompts

    ## subfunctions/config to process batch
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    # format prompts
    prompts =  dataset_to_prompt(ds,system_prompt)

    # vllm inference and extract output text
    start_id = 0
    vllm_out = []
    for batch_id, batch_prompts in enumerate(chunks(prompts, batch_size), start=start_id):
        currnt_id=batch_id*batch_size
        print(f"Processing {currnt_id}")
        outs = llm.generate(batch_prompts, sampling_params)
        for out in outs:
            text = out.outputs[0].text
            vllm_out.append(text)
        
    print("Inference completed!")

    # save raw output
    raw_output_path = os.path.join(
        data_dir,
        f"bprs_{model_id.split('/')[-1]}_raw_hf.jsonl",
    )
    raw_out=[]
    for out in vllm_out:
        raw_out.append({"output": out
        })

    with open(raw_output_path, "w", encoding="utf-8") as f:
        for record in raw_out:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Raw outputs saved to {raw_output_path}")


if __name__ == "__main__":
    main()
