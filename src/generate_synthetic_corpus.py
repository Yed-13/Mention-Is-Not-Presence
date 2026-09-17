#!/usr/bin/env python3
"""Expand the 10 contrast templates into a deterministic synthetic corpus.

Labels are inherited by construction, not presented as human annotations.
"""
import argparse, copy, json
from scriptbreak_eval import read_jsonl, write_jsonl

NAMES_A=list("林周陈许沈顾苏陆叶江")+["程安","方宁","孟夏","秦川","唐雨","宋乔","温言","夏禾","严冬","赵青"]
NAMES_B=["老周","老陈","老许","老沈","老顾","老苏","老陆","老叶","老江","老程","老方","老孟","老秦","老唐","老宋","老温","老夏","老严","老赵","老林"]
NAMES_C=["小林","小周","小陈","小许","小沈","小顾","小苏","小陆","小叶","小江","小程","小方","小孟","小秦","小唐","小宋","小温","小夏","小严","小赵"]
NAMES_D=["陈禾","周禾","许禾","沈禾","顾禾","苏禾","陆禾","叶禾","江禾","程禾","方禾","孟禾","秦禾","唐禾","宋禾","温禾","夏禾","严禾","赵禾","林禾"]
NAMES_E=["许宁","沈宁","顾宁","苏宁","陆宁","叶宁","江宁","程宁","方宁","孟宁","秦宁","唐宁","宋宁","温宁","夏宁","严宁","赵宁","林宁","周宁","陈宁"]
ITEM_A=["钥匙","门卡","戒指","录音笔","票根","印章","硬币","怀表","耳机","存储卡","药盒","信封","名片","徽章","手表","发卡","U盘","车钥匙","工作证","遥控器"]
ITEM_B=["剪刀","手电筒","药瓶","信封","螺丝刀","胶卷","梳子","记事本","尺子","胶带","钢笔","雨伞","水杯","地图","相机","口罩","纸袋","玩偶","纽扣","电池"]
CLOTH=["红色围巾","蓝色帽子","黑色外套","白色手套","黄色披肩","绿色领带","灰色毛衣","紫色头巾","棕色风衣","银色腰带","红色帽子","蓝色围巾","黑色手套","白色外套","黄色领带","绿色披肩","灰色头巾","紫色毛衣","棕色腰带","银色风衣"]
ANIMALS=["白马","黑狗","灰兔","花猫","白鹅","棕熊","红狐","梅花鹿","小羊","斑马","骆驼","孔雀","猎犬","奶牛","黑马","白猫","黄狗","松鼠","水牛","山羊"]
POSTER_ANIMALS=["黑猫","白狗","灰兔","红狐","白马","花鹿","小羊","孔雀","松鼠","黑狗","白猫","斑马","奶牛","黄狗","山羊","骆驼","白鹅","棕熊","水牛","猎犬"]

def replace_all(value, mapping):
    if isinstance(value,str):
        for old,new in mapping.items(): value=value.replace(old,new)
        return value
    if isinstance(value,list): return [replace_all(x,mapping) for x in value]
    if isinstance(value,dict): return {k:replace_all(v,mapping) for k,v in value.items()}
    return value

def main():
    p=argparse.ArgumentParser(); p.add_argument("--silver",required=True); p.add_argument("--out",required=True); p.add_argument("--variants",type=int,default=20); a=p.parse_args()
    if a.variants != 20: raise SystemExit("this frozen generator requires exactly 20 variants")
    base=read_jsonl(a.silver); by_family={}
    for r in base: by_family.setdefault(r["family_id"],[]).append(r)
    out=[]
    for fid, scenes in sorted(by_family.items()):
        for i in range(20):
            mapping={"林青":NAMES_A[i],"老周":NAMES_B[i],"小林":NAMES_C[i],"陈禾":NAMES_D[i],"许宁":NAMES_E[i]}
            if fid=="F01": mapping["钥匙"]=ITEM_A[i]
            if fid=="F03": mapping["剪刀"]=ITEM_B[i]
            if fid=="F04": mapping["红色围巾"]=CLOTH[i]
            if fid=="F07": mapping["白马"]=ANIMALS[i]
            if fid=="F10": mapping["黑猫"]=POSTER_ANIMALS[i]
            split="train" if i<12 else "dev" if i<16 else "test"
            for j, src in enumerate(sorted(scenes,key=lambda x:x["scene_id"]),1):
                r=replace_all(copy.deepcopy(src),mapping)
                r["scene_id"]=f"SYN_{fid}_{i:02d}_{j}"
                r["family_id"]=f"{fid}_{i:02d}"
                r["template_id"]=fid; r["variant_index"]=i; r["split"]=split
                r["status"]="approved_synthetic"; r["annotation_source"]="programmatic_by_construction"
                r["annotation_note"]="State inherited from a disclosed controlled template; not human annotation."
                out.append(r)
    write_jsonl(a.out,out)
    print(json.dumps({"scenes":len(out),"families":len({r['family_id'] for r in out}),"splits":{s:sum(r['split']==s for r in out) for s in ['train','dev','test']}},ensure_ascii=False))
if __name__=="__main__": main()
