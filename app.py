import torch
import streamlit as st
import pandas as pd
import numpy as np
from annotated_text import annotated_text
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer
import meta_question
from naver_clova_utils import *


def seed_everything(TORCH_SEED):
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


class CaseClassification:
    def __init__(self):
        CKPT_PATH = '../classification/best.pt'
        tokenizer = AutoTokenizer.from_pretrained(
            "monologg/kobert", use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            CKPT_PATH, num_labels=13)
        self.pipe = pipeline("text-classification", model=model,
                             tokenizer=tokenizer, return_all_scores=True)

    def __call__(self, text):
        pred = self.pipe(text)
        label_map = ['강제추행', '공무집행방해', '공연음란', '모욕', '무고', '사기', '상해', '업무방해', '위증', '준강간',
                     '특수협박', '폭행', '협박']
        pred_dict = {label_map[i]: v['score'] for i, v in enumerate(pred[0])}
        values = torch.tensor(list(pred_dict.values())).topk(2)[0]
        indexs = torch.tensor(list(pred_dict.values())).topk(2)[1]

        return {'label': indexs,
                'score': values}


class VictimClassification:
    def __init__(self):
        CKPT_PATH = '../classification/victim'
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CKPT_PATH)

    def __call__(self, text):
        # input_ids = self.tokenizer.encode(text, return_tensors='pt')
        input_ids = self.tokenizer.encode(
            text, return_tensors='pt', truncation=True, max_length=512)
        preds = self.model(input_ids)
        logits = preds.logits.detach().cpu()
        Softmax = torch.nn.Softmax(dim=1)
        logits = Softmax(logits)
        values = logits.topk(2)[0][0]
        indexs = logits.topk(2)[1][0]
        return {'label': indexs,
                'score': values}


st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.snow()
# st.title("🤖AI Legal Assistant🤖")
st.image("./asset/title.jpg")
case_clf = CaseClassification()
victim_clf = VictimClassification()

# col0, col1, col2, col3, col4 = st.columns([2, 2, 2, 2, 2])
col0, col1, col2, col3 = st.columns([2.5, 2.5, 2.5, 2.5])

with col0:
    case0 = st.button("SAMPLE1_모욕죄1")
with col1:
    case1 = st.button("SAMPLE1_모욕죄2")
with col2:
    case2 = st.button("SAMPLE2_사기죄1")
with col3:
    case3 = st.button("SAMPLE2_사기죄2")
# with col4:
#     case4 = st.button("SAMPLE3_상해죄")

col0, col1, col2 = st.columns([7, 1.5, 1.5])

with col0:
    if case0:
        case_text = """군대 성희롱
    2019년 8월 초 선임들이랑 동기들이 다 있는곳에서 한 선임이 갑자기 시마켄tv얘기를 하다가 저를 보면서 'ᄋᄋ이는 생긴게 X지를 잘빨게 생긴 상이노'라고 말했습니다. 그 말 한마디에 저를 제외한 다른 병들이낄낄대며 웃었고 저는 당황해서 어떻게 반응해야할지 몰랐습니다. 그래서 어버버..거리다 'ㅇㅇ이 X꼭지 빨고 있노'라고 제차 저를 희롱했습니다. 공개적인 장소에서 그런 모욕을 받았다는게 너무 수치스럽습니다..빨리 잊어야지 하고 다른생각도 하고 좀 더 제 일을 열심히 하려하지만 계속 머리속에서 그 문제의 선임병의 말과 다른 병들의 웃음이 떠나가지 않습니다. 어떻게 하는게 좋을지 모르겠습니다."""

    if case1:
        case_text = """지나가는 사람에게 성희롱한 놈들 대리신고 되나요?
대낮에 집앞 대로에서 씨발년 페미년 따먹어줄까 이런 소리 들리길래 내다봤더니 남자 둘이 길에 죽치고 앉아서 지나가는 여자한테 그런식으로 떠들고 있더라고요 완전 놀란게 바로 길 건너편에 유치원이 있는데 거기 앉아서 그러고 있었습니다 저만 들은게 아닐 듯 한데... 바로 밑이라서 한동안 보고 있었는데 아저씨들도 혼자 지나갔는데 거긴 아무 시비 안걸고 아기엄마들로 추정되는 여자들한테만 그런 욕설 지껄이면서 쳐다보며 낄낄낄낄... 무슨 못사는 나라에서 일하러 온 양아치들이 지들 말 못알아듣는 줄 알고 떠드는것처럼 그렇게 떠들어대는데 그런 놈들 생전 처음봤네요 요즘은 액면가만 가지고는 알 수가 없는데 일단 젊어뵈는 놈들이었습니다 사람들이 그냥 무시하고 지나가고 쳐다보고 지나가고 이러던데 제가 혹시나해서 녹음은 해놨습니다 근데 딱 그순간을 포착한건 아니라서 낄낄대는 부분만요 유치원 근처라서 cctv도 찾으면 있지 싶고요 저것들 안전지대로 여겨서 또 오면 어쩌나 싶은데 이거 제 3자가 신고 가능한지요?"""

    if case2:
        case_text = """사기당한 돈 받아내기
25만원이란 돈이 미성년자인 저에겐 너무나도 큰돈입니다. 지인에게 오토바이를 살려고 25만원을 줬어요. 알고보니 애초부터 사기를 칠려고 오토바이가 팔렸음에도 안팔렸다고 하며 저에게 25만원을 받아갔습니다. 입금 내역은 없으나 페이스북 메시지로 연락 오간 증거물은 있으나 물증이 되지않을것만같아 신고도 못하고 있습니다.메시지가 오간 계정은 그 지인이 계정을 비활성화 했는지 탈퇴했는지 연락이 못가게 되어있더군요.지인은 2명이고 둘다 성인입니다.치밀한것 같습니다연락은 받고 돈을 돌려주겠다 하지만 매번 기한을 미루기만하고 신고하던말던 배째라 형식으로 나올거 같습니다.매번 기다리는것도 힘들고 저도 필요할때 써야할때마다 돈이없어 힘들어요...신고를 한다면 어떻게 진행 하여야하고,돈은 꼭 받아내고 싶습니다.그 지인이 다른곳에도도 돈빌린게많아 제돈은 무시하는것 같습니다.말만 갚는다하지 마음도 없어보이구요...도와주세요"""

    if case3:
        case_text = """사기 당했습니다 도와주시길바라겠습니다
안녕하세요 항상 고생많으십니다. 다름이아니라 제가 1월경에 물건을 거래하기로하였고 100만원을 입금하였습니다. 그런데 확인해보니 그 제품이 가품인데 진품인 거처럼 속여서 판거더라구요. 그래서 거래취소를 요청하고 돈을 돌려달라하였습니다. 하지만 6월이 될때까지 돈을 돌려주지 않고있습니다. 사정이 있다 하시고 준다고하여 믿고 기다렸으나 현재는 카톡도 삭제되었고 번호또한 없는 번호라 나왔습니다. 번개장터 거래글은 현재 삭제되었으며 입금기록, 카톡대화내용등은 남아있습니다. 어찌하면 좋을지 여쭈어볼라고 질문 올립니다현재 군복무중이며 경찰서를 방문하지 않고 신고할수 있는 방법이 있을까요? 항상 수고많으시고 감기조심하세요~"""

#     if case4:
#         case_text = """가정폭력 신고 접수
# 18살 여자입니다 어렸을 때부터 아빠가 폭력을 많이 휘둘렀습니다 첫 기억인 5살 때부터 꾸준히 폭력과 질나쁜 언행이 수없이 오갔습니다 하지만 두 분이 재혼이셨고 아빠 때문에 신용불량 위기에 처한 엄마는 두 번 실수하기 싫고 손해본 게 너무 많아서 이혼은 못 하겠다 해서 경찰에 신고 한 번 안 하고 살았습니다 그런데 2년 전 너무 심한 폭행에 경찰에 신고를 했고 경찰서에 갔지만 초범이기 때문에 접근금지명령만 떨어졌고 그마저도 엄마는 풀어버리고 집에서 같이 살았습니다 그 이후에는 몸 싸움이 일어나면 구속이 되니깐 아빠가 욕이랑 말 싸움만 했어서 큰 일은 없었는데 오늘 엄마 아빠랑 아빠 친구분이랑 시골에 내려가서 말 싸움이 있었고 엄마가 목을 졸랐습니다 저희 아빠가 힘이 너무 세셔서 보복을 하면 친구분이 못 말릴 거 같아서 경찰에 신고하셨다고 하고 이번엔 엄마가 폭행죄가 성립이 된 거 같습니다 제가 예전에 있었던 일을 증언해야 된다는데 만약 제가 가정폭력을 다 얘기하면 강제로도 이혼이 될까요? 제발 벗어나고 싶어요."""

    # if case0 or case1 or case2 or case3 or case4:
    if case0 or case1 or case2 or case3:
        items = st.text_area(
            '질문을 입력하세요', case_text
        )
    else:
        items = st.text_area(
            '질문을 입력하세요',
        )
    # submission = st.button("질문하기")

if len(items) > 0:
    case_pred = case_clf(items)

    victim_pred = victim_clf(items)

    case_label_map = ['강제추행', '공무집행방해', '공연음란', '모욕', '무고', '사기', '상해', '업무방해', '위증', '준강간',
                      '특수협박', '폭행', '협박']
    case_label_1 = case_label_map[case_pred['label'][0].item()]
    case_label_2 = case_label_map[case_pred['label'][1].item()]
    case_value_1 = round(case_pred['score'][0].item()*100, 1)
    case_value_2 = round(case_pred['score'][1].item()*100, 1)

    with col1:
        delta_text = str(round(case_pred['score'][0].item(
        )*100 - case_pred['score'][1].item()*100, 2)) + "%"
        st.metric(label=case_label_1,
                  value=f"{case_value_1}%", delta=delta_text)
        st.metric(label=case_label_2,
                  value=f"{case_value_2}%")

    victim_label_map = ["가해", "피해"]
    victim_label_1 = victim_label_map[victim_pred['label'][0].item()]
    victim_label_2 = victim_label_map[victim_pred['label'][1].item()]
    victim_value_1 = round(victim_pred['score'][0].item() * 100, 1)
    victim_value_2 = round(victim_pred['score'][1].item() * 100, 1)

    if victim_label_1 == "피해":
        if case_label_1 in list(meta_question.VICTIM_HUMAN.keys()):
            question_lst = meta_question.VICTIM_HUMAN[case_label_1]
        else:
            question_lst = meta_question.VICTIM_MACHIN[case_label_1]
    else:
        if case_label_1 in list(meta_question.PERPETRATOR_HUMAN.keys()):
            question_lst = meta_question.PERPETRATOR_HUMAN[case_label_1]
        else:
            question_lst = meta_question.PERPETRATOR_MACHIN[case_label_1]

    with col2:
        delta_text = str(round(victim_pred['score'][0].item(
        ) * 100 - victim_pred['score'][1].item() * 100, 2)) + "%"
        st.metric(label=victim_label_1,
                  value=f"{victim_value_1}%", delta=delta_text)
        st.metric(label=victim_label_2,
                  value=f"{victim_value_2}%")

    col1, col2 = st.columns([5, 5])
    with col1:
        # st.write("[질문자 화면]")
        st.image("./asset/phase1.png")
        # st.write("원활한 법률자문을 위하여 아래 질문을 읽고, 답변 작성 후 제출해주세요.")
        question_dict = {q: "" for q in question_lst}
        for k, v in question_dict.items():
            question_dict[k] = st.text_input(k, v)
            st.write(question_dict[k])
        print(question_dict)
        st.button('추가 질문하기')

    with col2:
        # st.write("[법률가 화면]")
        st.image("./asset/phase2.png")

        st.write("법률자문 초안")
        answer = ''
        while (len(answer) < 100):
            answer = answer_generation(items)
            if '법무법인' in answer:
                answer = ''
        st.caption(answer)
        st.write("\n")

        requisite = meta_question.comp_dict[case_label_1]
        requisite_dict = dict()
        for req in requisite:
            print(req)
            try:
                start_time = time.time()
                requisite_dict[req] = requisite_extraction(
                    items, case_label_1, req)
                if time.time() - start_time > 20:
                    continue
            except:
                continue
        print(requisite_dict)
        # http://www.n2n.pe.kr/lev-1/color.htm
        colors = ["rgb(255, 255, 204)",
                  "rgb(204, 255, 204)", "rgb(204, 255, 255)"]
        for i, (k, v) in enumerate(requisite_dict.items()):
            if k != None and v != None:
                annotated_text((v, k, colors[i]))

        st.write("\n")

        answer = st.text_area("답변을 입력하세요.")
        st.button('답변하기')
