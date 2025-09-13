import streamlit as st
from PIL import Image
from streamlit_extras.let_it_rain import rain

# ou plus précis
from my_mirror_on_cloud.streamlit_pipeline import (
    init_model,
    search_recommended_outfit,
)

from scripts.load_image import get_url_from_image_path, load_image_from_url

from loguru import logger
import tempfile

from my_mirror_on_cloud.catvton_caller import process_images_catvton


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


def save_cloth_images(outfit):
    images = st.session_state.recommended_outfit[outfit]["cloth_path"]
    for path in images:
        image_url = get_url_from_image_path(path)
        img = load_image_from_url(image_url)  # assumes PIL Image returned
        img.save("cloth.jpg")
        logger.info("Saved cloth image to cloth.jpg")


def show():
    success = st.empty()
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

        # My profile
        columns = st.columns(2)
        with columns[0]:
            expander = st.expander("My profile", expanded=False)
            age = expander.slider("Age", min_value=0, max_value=120, value=25)
            gender = expander.selectbox("Gender", options=["Male", "Female", "Other"])
        with columns[1]:
            expander = st.expander("Your picture", expanded=False)
            st.session_state.uploaded_file = expander.file_uploader(
                "Upload an image of you today", type=["png", "jpg", "jpeg"]
            )

            # Display image placeholder initially
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
                    Waiting for image
                </div>
                """,
                unsafe_allow_html=True,
            )

            # When user uploads a file
            if st.session_state.uploaded_file:
                # Show spinner during image loading
                with st.spinner("Uploading image... ⏳"):
                    st.session_state.image = Image.open(st.session_state.uploaded_file)

                # Then replace placeholder with the image
                img.image(st.session_state.image, width=300)

                st.success("Image uploaded successfully! ✅")
                # Save the uploaded person image temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".png"
                ) as tmp_person:
                    st.session_state.image.save(tmp_person.name)
                    person_img_path = tmp_person.name
                    logger.info(
                        f"Saved uploaded person image to temporary file: {person_img_path}"
                    )

        submitted = st.form_submit_button("Submit")

        if submitted:
            st.session_state.show_outfits = True
            st.session_state.outfit_choice = None
            success = st.success(
                "Profile information submitted successfully!", icon="✅"
            )
            # Launch search_recommended_outfit
            st.session_state.recommended_outfit = search_recommended_outfit(
                f"{query}. I'm a {gender} and I am {age}",
                person_img_path if st.session_state.image is not None else None,
                st.session_state.init_model,
                st.session_state.type_of_query,
            )

            with st.expander("Returned data"):
                st.write(st.session_state.recommended_outfit)

    # Display clothes
    if st.session_state.show_outfits:
        if st.session_state.outfit_choice is None:
            if st.session_state.recommended_outfit == "Need picture":
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
                    outfit_choice = st.radio("Select an outfit", options=options)
                    submitted_choice = st.form_submit_button("Submit")

                    if submitted_choice:
                        save_cloth_images(options.index(outfit_choice))

                        if st.session_state.image is not None:
                            try:
                                cloth_img_path = "cloth.jpg"

                                try:
                                    with st.spinner(
                                        "Switching clothes on your picture... ⏳"
                                    ):
                                        output_img_path = "output.jpg"
                                        process_result = process_images_catvton(
                                            person_img_path,
                                            cloth_img_path,
                                            output_img_path,
                                        )
                                    if process_result.get("success"):
                                        st.image(
                                            output_img_path,
                                            caption="Your new outfit",
                                            width=300,
                                        )
                                    else:
                                        st.error(
                                            f"Failed to process images at step: {process_result.get('step', 'unknown')}: "
                                            f"{process_result.get('response', '')}"
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Exception during image processing: {e}"
                                    )
                                    st.error(
                                        "Image processing failed due to a server or network error. "
                                        "Please try again later."
                                    )

                            except Exception as e:
                                logger.error(f"Error during processing images: {e}")
                                st.error(f"An error occurred: {e}")
                        else:
                            st.warning("Add a picture of you")

            else:
                st.warning("Request need clarification")
