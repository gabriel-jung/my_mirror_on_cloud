# import streamlit as st
# from PIL import Image
# import time
# from my_mirror_on_cloud.streamlit_pipeline import init_model, search_recommended_outfit


# def init_state():
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {
#                 "role": "assistant",
#                 "content": "👋 Welcome to My Mirror on Cloud! Upload your picture and I'll help you find the perfect outfit.",
#             }
#         ]
#         st.session_state.chat_step = 1
#         st.session_state.user_data = {}
#         collections = init_model()
#         st.session_state.tenues_collection = collections[0]
#         st.session_state.clothes_collection = collections[1]
#         st.session_state.catalogue_collection = collections[2]
#         st.session_state.model_lang = collections[3]
#         st.session_state.tokenizer_lang = collections[4]


# def chatbot_flow(user_input=None, uploaded_file=None):
#     step = st.session_state.chat_step
#     reply = None

#     # --- Étape 1 : Upload photo ---
#     if step == 1:
#         if uploaded_file:
#             image = Image.open(uploaded_file)
#             st.session_state.user_data["photo"] = image
#             reply = "Nice photo! 😊 Now, tell me your age."
#             st.session_state.chat_step = 2

#     # --- Étape 2 : Age ---
#     elif step == 2 and user_input:
#         st.session_state.user_data["age"] = user_input
#         reply = "Got it! What’s your gender? (Male / Female / Other)"
#         st.session_state.chat_step = 3

#     # --- Étape 3 : Gender ---
#     elif step == 3 and user_input:
#         st.session_state.user_data["gender"] = user_input
#         reply = "Perfect 👍 Now, describe the outfit or occasion you’re looking for."
#         st.session_state.chat_step = 4

#     # --- Étape 4 : Query outfit ---
#     elif step == 4 and user_input:
#         st.session_state.user_data["query"] = user_input
#         reply = "Thanks! Let me find some outfits for you... ⏳"
#         st.session_state.recommended_outfit = search_recommended_outfit(
#             user_input,
#             st.session_state.tenues_collection,
#             st.session_state.clothes_collection,
#             st.session_state.catalogue_collection,
#             st.session_state.model_lang,
#             st.session_state.tokenizer_lang,
#         )

#         st.markdown("Here are 3 outfit options! 👗👔")
#         options = ["Outfit 1", "Outfit 2", "Outfit 3"]
#         st.write(st.session_state.recommended_outfit)
#         for i, outfit in enumerate(st.session_state.recommended_outfit):
#             st.image("https://via.placeholder.com/150", caption=options[i])
#         reply = "Please type: Outfit 1, Outfit 2, or Outfit 3."
#         st.session_state.chat_step = 6

#     # --- Étape 6 : Choix final ---
#     elif step == 6 and user_input:
#         choice = user_input.strip()
#         if choice in ["Outfit 1", "Outfit 2", "Outfit 3"]:
#             idx = ["Outfit 1", "Outfit 2", "Outfit 3"].index(choice)
#             outfit = st.session_state.recommended_outfit[idx]
#             st.session_state.user_data["choice"] = outfit["cloth_path"]

#             with st.chat_message("assistant"):
#                 st.markdown(f"You chose **{choice}** 🎉 Here’s how it looks on you...")
#                 time.sleep(2)
#                 st.image("https://via.placeholder.com/200", caption="Your new outfit")

#             reply = None  # pas besoin de message texte en plus
#             st.session_state.chat_step = 7
#         else:
#             reply = "Please choose Outfit 1, Outfit 2, or Outfit 3."

#     return reply


# def show_chatbot():
#     st.title("🪞 My Mirror on Cloud - Chatbot Edition")

#     col1, col2 = st.columns([1, 2], border=True)

#     with col1:
#         picture = st.camera_input("Take a picture", disabled=False)

#         # display the image placeholder (bordered box) define size
#         img = st.empty()
#         img.markdown(
#             """
#             <div style="
#                 width:200px;
#                 height:200px;
#                 border:2px dashed #888;
#                 display:flex;
#                 align-items:center;
#                 justify-content:center;
#                 color:#888;
#                 border-radius:10px;
#             ">
#                 En attente d'image
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#         if picture:
#             # Affichage immédiat de l'image
#             image = Image.open(picture)
#             img.image(image, caption="Image uploadée", width=200)

#             # --- Section d'analyse ---
#             with st.spinner("Analyse en cours... ⏳"):
#                 time.sleep(2)

#     with col2:
#         # Formulaire profil utilisateur
#         st.subheader("Tell us more about you!")

#         with st.form("user_profile_form2"):
#             with st.expander("Your profile"):
#                 age = st.slider("Age", min_value=0, max_value=120, value=25)
#                 gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
#                 submitted = st.form_submit_button("Submit")

#         init_state()

#         # --- Display past messages ---
#         for msg in st.session_state.messages:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         # --- Upload pour étape 1 ---
#         if st.session_state.chat_step == 1:
#             uploaded_file = st.file_uploader(
#                 "Upload your picture", type=["png", "jpg", "jpeg"]
#             )
#             if uploaded_file:
#                 reply = chatbot_flow(uploaded_file=uploaded_file)
#                 if reply:
#                     st.session_state.messages.append(
#                         {"role": "assistant", "content": reply}
#                     )
#                     st.rerun()

#         # --- Chat input ---
#         if st.session_state.chat_step > 1:
#             if prompt := st.chat_input("Your answer..."):
#                 st.session_state.messages.append({"role": "user", "content": prompt})
#                 reply = chatbot_flow(user_input=prompt)
#                 if reply:
#                     st.session_state.messages.append(
#                         {"role": "assistant", "content": reply}
#                     )
#                     st.rerun()
