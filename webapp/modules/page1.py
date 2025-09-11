import streamlit as st
from PIL import Image
import time

# ou plus précis
from my_mirror_on_cloud.streamlit_pipeline import (
    init_model,
    search_recommended_outfit,
)

from scripts.load_image import get_url_from_image_path, load_image_from_url


def show():
    if "show_outfits" not in st.session_state:
        st.session_state.show_outfits = False
    if "outfit_choice" not in st.session_state:
        st.session_state.outfit_choice = None
    if "init_model" not in st.session_state:
        st.session_state.init_model = init_model()
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    st.title("**Welcome to My Mirror on Cloud!**")
    st.subheader("This application recommends you outfits based on your requests")
    st.write("Use the tabs above to learn more about the app.")

    # taille des colonnes
    col1, col2 = st.columns([1, 2], border=True)

    with col1:
        st.session_state.uploaded_file = st.file_uploader(
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

        if st.session_state.uploaded_file:
            # Affichage immédiat de l'image
            image = Image.open(st.session_state.uploaded_file)
            img.image(image, caption="Image uploadée", width=200)

            # --- Section d'analyse ---
            with st.spinner("Analyse en cours... ⏳"):
                time.sleep(1)

    with col2:
        # Formulaire profil utilisateur
        st.subheader("Tell us more about you!")

        with st.form("user_profile_form"):
            with st.expander("Your profile"):
                # age = st.slider("Age", min_value=0, max_value=120, value=25)
                gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

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
                st.session_state.outfit_choice = None
                success = st.success(
                    "Profile information submitted successfully!", icon="✅"
                )
                # Launch search_recommended_outfit
                st.session_state.recommended_outfit = search_recommended_outfit(
                    f"{query}. I'm a {gender}.", image, st.session_state.init_model
                )
                with st.expander("Afficher l'array", expanded=False):
                    st.write(st.session_state.recommended_outfit)

    if st.session_state.show_outfits:
        if st.session_state.outfit_choice is None:
            if st.session_state.recommended_outfit is not None:
                with st.spinner("Finding the perfect outfit for you... ⏳"):
                    st.balloons()
                    success.empty()

                st.success("Outfit recommendations are ready!")
                col1, col2, col3 = st.columns([1, 1, 1], border=True)
                options = ["Outfit 1", "Outfit 2", "Outfit 3"]
                with col1:
                    st.subheader(options[0])
                    images = st.session_state.recommended_outfit[0]["cloth_path"]
                    cols = st.columns(2)
                    for i, path in enumerate(images):  # max 3 images
                        image_url = get_url_from_image_path(path)
                        img = load_image_from_url(image_url)

                        # Choisir la colonne de manière alternée
                        col = cols[i % 2]

                        # Afficher l'image
                        col.image(img, channels="RGB")
                with col2:
                    st.subheader(options[1])
                    images = st.session_state.recommended_outfit[1]["cloth_path"]
                    cols = st.columns(2)

                    for i, path in enumerate(images):  # max 3 images
                        image_url = get_url_from_image_path(path)
                        img = load_image_from_url(image_url)

                        # Choisir la colonne de manière alternée
                        col = cols[i % 2]

                        # Afficher l'image
                        col.image(img, channels="RGB")
                with col3:
                    st.subheader(options[2])
                    images = st.session_state.recommended_outfit[2]["cloth_path"]
                    cols = st.columns(2)

                    for i, path in enumerate(images):  # max 3 images
                        image_url = get_url_from_image_path(path)
                        img = load_image_from_url(image_url)

                        # Choisir la colonne de manière alternée
                        col = cols[i % 2]

                        # Afficher l'image
                        col.image(img, channels="RGB")

                # ask choose one outfit
                st.markdown("### Choose your favorite outfit:")
                with st.form("choice_form"):
                    outfit_choice = st.radio(
                        "Select an outfit", options=["Outfit 1", "Outfit 2", "Outfit 3"]
                    )
                    st.session_state.outfit_choice = [
                        outfit_choice,
                        st.session_state.recommended_outfit[
                            options.index(outfit_choice)
                        ]["cloth_path"],
                    ]
                    submitted_choice = st.form_submit_button("Submit")

                    if submitted_choice:
                        st.success(f"You have selected {outfit_choice}!")
                        st.write(
                            st.session_state.outfit_choice[0],
                            st.session_state.outfit_choice[1],
                        )
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

            else:
                st.warning("Request need clarification")
