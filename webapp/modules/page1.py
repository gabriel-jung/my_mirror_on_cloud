import streamlit as st
from PIL import Image
from streamlit_extras.let_it_rain import rain

# ou plus précis
from my_mirror_on_cloud.streamlit_pipeline import (
    init_model,
    search_recommended_outfit,
)

from scripts.load_image import get_url_from_image_path, load_image_from_url


def rain_clothes():
    rain(
        emoji="👗     👕",
        font_size=84,
        falling_speed=10,
        animation_length=1,
    )

def display_clothes(outfit):
    options = ["Outfit 1", "Outfit 2", "Outfit 3"]
    st.subheader(options[outfit])
    images = st.session_state.recommended_outfit[outfit]["cloth_path"]
    cols = st.columns(2)
    for i, path in enumerate(images):  # max 3 images
        image_url = get_url_from_image_path(path)
        img = load_image_from_url(image_url)
        col = cols[i % 2]
        col.image(img, channels="RGB")

def rain_clothes():
    rain(
        emoji="👗     👕",
        font_size=84,
        falling_speed=10,
        animation_length=1,
    )

def display_clothes(outfit):
    options = ["Outfit 1", "Outfit 2", "Outfit 3"]
    st.subheader(options[outfit])
    images = st.session_state.recommended_outfit[outfit]["cloth_path"]
    cols = st.columns(2)
    for i, path in enumerate(images):  # max 3 images
        image_url = get_url_from_image_path(path)
        img = load_image_from_url(image_url)
        col = cols[i % 2]
        col.image(img, channels="RGB")

def show():
    if "show_outfits" not in st.session_state:
        st.session_state.show_outfits = False
    if "outfit_choice" not in st.session_state:
        st.session_state.outfit_choice = None
    if "init_model" not in st.session_state:
        st.session_state.init_model = init_model()
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.image = None
    if "recommended_outfit" not in st.session_state:
        st.session_state.recommended_outfit = None

    st.title("**Welcome to My Mirror on Cloud!**")
    st.subheader("This application recommends you outfits based on your requests")

    with st.form("user_profile_form"):           
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

        # My profile
        expander = st.expander("My profile")
        age = expander.slider("Age", min_value=0, max_value=120, value=25)
        gender = expander.selectbox("Gender", options=["Male", "Female", "Other"])
        st.session_state.uploaded_file = expander.file_uploader(
            "Upload an image of you today", type=["png", "jpg", "jpeg"]
        )

        # display the image placeholder (bordered box) define size
        img = expander.empty()
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
            st.session_state.image = Image.open(st.session_state.uploaded_file)
            img.image(st.session_state.image, caption="Image uploadée", width=200)


        if submitted:
            st.session_state.show_outfits = True
            st.session_state.outfit_choice = None
            success = st.success("Profile information submitted successfully!", icon="✅")
            # Launch search_recommended_outfit
            st.session_state.recommended_outfit = search_recommended_outfit(f"{query}. I'm a {gender} and I am {age}", st.session_state.image, st.session_state.init_model, st.session_state.type_of_query)
            
            with st.expander("Returned data"):
                st.write(st.session_state.recommended_outfit)

    
    # Display clothes
    if st.session_state.show_outfits:
        if st.session_state.outfit_choice is None:
            if  st.session_state.recommended_outfit == "Need picture":
                st.warning("Add a picture of you")
                st.session_state.recommended_outfit = None
            if st.session_state.recommended_outfit is not None:
                with st.spinner("Finding the perfect outfit for you... ⏳"):  
                    rain_clothes()
                    success.empty()

                st.success("Outfit recommendations are ready!")
                col1, col2, col3 = st.columns([1, 1, 1], border=True)
                
                with col1:
                    display_clothes(0)
                with col2:
                    display_clothes(1)
                with col3:
                    display_clothes(2)

                # ask choose one outfit
                st.markdown("### Choose your favorite outfit:")
                options = ["Outfit 1", "Outfit 2", "Outfit 3"]
                with st.form("choice_form"):
                    outfit_choice = st.radio(
                        "Select an outfit", options=options
                    )                    
                    st.session_state.outfit_choice = [outfit_choice, st.session_state.recommended_outfit[options.index(outfit_choice)]["cloth_path"]]
                    submitted_choice = st.form_submit_button("Submit")
                    if st.session_state.image is not None:
                        if submitted_choice:
                            st.success(f"You have selected {outfit_choice}!")
                            st.write(st.session_state.outfit_choice[0], st.session_state.outfit_choice[1])
                                                    
                            with st.spinner("Switching clothes on your picture... ⏳"):
                                img.image(
                                    "https://via.placeholder.com/200",
                                    caption="Your new outfit",
                                    width=200,
                                )
                    else:
                        st.warning("Add a picture of you")
                
            else:
                st.warning("Request need clarification")
