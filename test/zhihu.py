import requests

"""
session_token: ba53a088a887d201384d46472426d134
desktop: true
page_number: 6
limit: 6
action: down
after_id: 29

session_token: ba53a088a887d201384d46472426d134
desktop: true
page_number: 5
limit: 6
action: down
after_id: 23
"""

headers = {
'accept':'*/*',
'accept-encoding':'gzip,deflate,br',
'accept-language':'zh-CN,zh;q=0.9',
'cookie':'_xsrf=qGjxJErDQqQF5MBvIq2pKMeX1h3t534L;_zap=50cae285-d7cc-4041-a93d-ca93a906e0ed;d_c0="AODuZYJVrw-PToYiiTjKLX2IY3teWTQt0QU=|1562244504";q_c1=cddee62f8100410abd367b816e3110cd|1562244523000|1562244523000;__utma=51854390.2122965702.1562244526.1562244526.1562244526.1;__utmz=51854390.1562244526.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic;__utmv=51854390.000--|3=entry_date=20190704=1;tgw_l7_route=66cb16bc7f45da64562a077714739c11;Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1569640073;capsion_ticket="2|1:0|10:1569640073|14:capsion_ticket|44:Y2MwMmM5ZTVlZGIxNDcxMmFjOWFhMGJmNTQwMGMzMWE=|da14643428f78e420c3bfcb395a08570fc9a77c4153ea6f6e351516b0e45123b";z_c0="2|1:0|10:1569640097|4:z_c0|92:Mi4xdmhTb0VnQUFBQUFBNE81bGdsV3ZEeVlBQUFCZ0FsVk5vUng4WGdBWFJtcFFiVXNiRnFCOGVJNXp0bkJjUWlfT2d3|77b7a9d8e2a58cd1710940dfe00c7c0ae14ca3ecacd11302ca84057e4d075b9f";unlock_ticket="APDsUZaEHBAmAAAAYAJVTanVjl1aPFj-6iiotiCmAWWBOE8ivo930g==";tst=r;Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1569640099',
'referer':'https://www.zhihu.com/',
'sec-fetch-mode':'cors',
'sec-fetch-site':'same-origin',
'user-agent':'Mozilla/5.0(WindowsNT10.0;Win64;x64)AppleWebKit/537.36(KHTML,likeGecko)Chrome/76.0.3809.100Safari/537.36',
'x-ab-param':'se_aa_base=1;se_college=default;tsp_childbillboard=2;zr_km_topic_zann=new;se_mclick=0;ug_fw_answ_aut_1=0;top_v_album=1;soc_special=0;se_backsearch=0;tp_topic_head=0;zr_infinity_member=close;zr_video_recall=current_recall;tp_header_style=1;pf_newguide_vertical=0;li_search_answer=2;zr_km_item_prerank=old;li_se_xgb=0;zr_answer_rec_cp=open;zr_km_item_cf=open;se_go_ztext=0;se_site_onebox=0;se_lottery=0;li_qa_new_cover=0;soc_bigone=1;top_universalebook=1;top_test_4_liguangyi=1;se_college_cm=1;se_ltr_user=1;se_cardrank_2=0;se_likebutton=0;se_famous=1;zr_km_style=base;ug_follow_topic_1=2;zr_ans_rec=gbrank;zr_intervene=0;se_search_feed=N;tsp_billboardsheep2=2;li_tjys_ec_ab=0;zr_article_new=close;se_featured=1;top_hotcommerce=1;ls_videoad=2;li_back=1;li_se_heat=0;se_payconsult=5;top_new_feed=5;soc_bignew=1;ls_fmp4=0;se_subtext=1;se_cardrank_3=0;zr_km_feed_nlp=old;zr_km_recall=default;zr_video_rank_nn=new_rank;se_agency=0;ug_follow_answerer=0;li_se_section=0;zr_item_nn_recall=close;se_com_boost=2;ug_newtag=1;top_ydyq=X;zr_art_rec=base;zr_km_feed_prerank=new;se_zu_onebox=0;li_se_paid_answer=0;zr_slot_cold_start=aver;zr_man_intervene=0;se_time_threshold=0;li_video_section=0;li_vip_no_ad_mon=0;se_ctr_pyc=0;li_qa_cover=old;li_se_vertical=1;se_wannasearch=a;tp_m_intro_re_topic=1;soc_notification=1;pf_creator_card=1;se_ios_spb309=1;se_hotsearch=1;zw_sameq_sorce=999;se_cardrank_1=0;se_ctr_user=1;se_cardrank_4=0;tp_sft_v2=d;pf_fuceng=1;qap_payc_invite=0;zw_payc_qaedit=0;pf_noti_entry_num=0;se_dnn_mt=1;se_mclick1=2;se_topiclabel=1;se_perf=1;top_ebook=0;tsp_vote=2;se_expired_ob=0;se_webtimebox=1;zr_search_xgb=1;zr_video_rank=new_rank;se_dnn_unbias=0;se_p_slideshow=0;se_ad_index=10;se_zu_recommend=0;sem_up_growth=in_app;li_se_kv=0;se_use_zitem=0;tp_qa_toast=1;top_quality=0;top_root=0;top_vipconsume=1;tsp_hotctr=2;se_auto_syn=0;se_movietab=1;soc_zuichangfangwen=2;se_webmajorob=0;se_qua_boost=0;tp_sft=a;ug_zero_follow_0=0;zr_rec_answer_cp=open;se_webrs=1;se_new_topic=0;se_hot_timebox=1;zr_rel_search=base;zr_test_aa1=1;li_pay_banner_type=0;li_android_vip=0;se_col_boost=0;se_spb309=0;se_mctr=0;se_mobileweb=1;tsp_newchild=4;ug_zero_follow=0;ug_follow_answerer_0=0;ls_new_upload=1;zr_cold_start=0;se_preset_tech=0;ls_zvideo_license=0;tp_sticky_android=2;se_websearch=3;se_adxtest=1;tp_qa_metacard=1;soc_update=0;li_hot_score_ab=0;zr_km_slot_style=event_card;tp_meta_card=0;li_purchase_test=0;li_album_liutongab=0;se_whitelist=1;se_billboardsearch=0;se_amovietab=1;tp_club_qa=2;pf_foltopic_usernum=50;li_se_album_card=0;se_ltr_dnn_cp=0;zr_km_answer=open_cvr;se_colorfultab=1;se_ctr_topic=0;se_waterfall=0;tp_qa_metacard_top=top;top_native_answer=1;ug_goodcomment_0=1;zr_km_prerank=new',
'x-api-version':'3.0.53',
'x-requested-with':'fetch',
}

url = 'https://www.zhihu.com/api/v3/feed/topstory/recommend?session_token=ba53a088a887d201384d46472426d134&desktop=true&page_number=5&limit=6&action=down&after_id=23'
res = requests.get(url, headers=headers)
print()