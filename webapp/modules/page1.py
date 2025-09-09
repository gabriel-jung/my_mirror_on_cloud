import streamlit as st
from PIL import Image
import time

# ou plus précis
from src.my_mirror_on_cloud.streamlit_pipeline import (
    init_model,
    search_recommended_outfit,
)


def show():
    if "show_outfits" not in st.session_state:
        st.session_state.show_outfits = False
    if "outfit_choice" not in st.session_state:
        st.session_state.outfit_choice = None
    if "tenues_collection" not in st.session_state:
        collections = init_model()
        st.session_state.tenues_collection = collections[0]
        st.session_state.clothes_collection = collections[1]
        st.session_state.model_lang = collections[2]
        st.session_state.tokenizer_lang = collections[3]

    st.title("**Welcome to My Mirror on Cloud!**")
    st.subheader("This application recommends you outfits based on your requests")
    st.write("Use the tabs above to learn more about the app.")

    # taille des colonnes
    col1, col2 = st.columns([1, 2], border=True)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image of you today", type=["png", "jpg", "jpeg"]
        )

        # display the image placeholder (bordered box) define size
        img = st.empty()
        img.markdown(
            """
            <div style="
                width:200px;
                height:200px;
                border:2px dashed #888;
                display:flex;
                align-items:center;
                justify-content:center;
                color:#888;
                border-radius:10px;
            ">
                En attente d'image
            </div>
            """,
            unsafe_allow_html=True,
        )

        if uploaded_file:
            # Affichage immédiat de l'image
            image = Image.open(uploaded_file)
            img.image(image, caption="Image uploadée", width=200)

            # --- Section d'analyse ---
            with st.spinner("Analyse en cours... ⏳"):
                time.sleep(2)

    with col2:
        # Formulaire profil utilisateur
        st.subheader("Tell us more about you!")

        with st.form("user_profile_form"):
            # with st.expander("Your profile"):
            # age = st.slider("Age", min_value=0, max_value=120, value=25)
            # gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

            st.markdown(
                """
                <div style="display:flex; align-items:center;">
                    <h3 style="margin:0;">What are you looking for?</h3>
                    <span title="Exemple :
                    - Casual outfit for a day out with friends
                    - Formal attire for a wedding
                    - I'm looking for a summer dress for a beach vacation
                    - Do you have red top and blue jeans in your collection?"
                        style="margin-left:8px; cursor:help; font-size:18px;">ℹ️</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            query = st.text_area("Describe your desired outfit or occasion", height=100)

            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.show_outfits = True
                success = st.success(
                    "Profile information submitted successfully!", icon="✅"
                )
                # Launch search_recommended_outfit
                result = search_recommended_outfit(
                    query,
                    st.session_state.tenues_collection,
                    st.session_state.model_lang,
                    st.session_state.tokenizer_lang,
                )
                st.write(result)

    if st.session_state.show_outfits:
        if st.session_state.outfit_choice is None:
            with st.spinner("Finding the perfect outfit for you... ⏳"):
                time.sleep(3)
                st.balloons()
                success.empty()

        st.success("Outfit recommendations are ready!")
        col1, col2, col3 = st.columns([1, 1, 1], border=True)
        with col1:
            st.image("https://via.placeholder.com/150", caption="Outfit 1")
        with col2:
            st.image("https://via.placeholder.com/150", caption="Outfit 2")
        with col3:
            st.image("https://via.placeholder.com/150", caption="Outfit 3")

        # ask choose one outfit
        st.markdown("### Choose your favorite outfit:")
        with st.form("choice_form"):
            outfit_choice = st.radio(
                "Select an outfit", options=["Outfit 1", "Outfit 2", "Outfit 3"]
            )
            st.session_state.outfit_choice = outfit_choice
            submitted_choice = st.form_submit_button("Submit")

            if submitted_choice:
                st.success(f"You have selected {outfit_choice}!")
                img.markdown(
                    """
                    <div style="
                        width:200px;
                        height:200px;
                        border:2px dashed #888;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        color:#888;
                        border-radius:10px;
                    ">
                        Loading image...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.spinner("Switching clothes on your picture... ⏳"):
                    time.sleep(3)
                    img.image(
                        "https://via.placeholder.com/200",
                        caption="Your new outfit",
                        width=200,
                    )
