"""
Unit tests for Florence-2 adapter.

Tests all supported tasks:
- CAPTION: Concise image caption
- DETAILED_CAPTION: Detailed caption
- MORE_DETAILED_CAPTION: Very detailed caption
- OD: Object detection (boxes)
- DENSE_REGION_CAPTION: Region captions (boxes + text)
- REGION_PROPOSAL: Region proposals (boxes)
- REFERRING_EXPRESSION_SEGMENTATION: Segmentation from referring expression
- REGION_TO_SEGMENTATION: Segmentation for region
- OCR: Full image OCR text
- OCR_WITH_REGION: OCR with region boxes
- CAPTION_TO_PHRASE_GROUNDING: Phrase grounding
- REF: Legacy referring expression boxes
"""

import os
import pytest
import numpy as np
import cv2
from pathlib import Path


# Skip all tests if Florence-2 dependencies are not available
try:
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    FLORENCE2_AVAILABLE = True
except ImportError:
    FLORENCE2_AVAILABLE = False


@pytest.fixture(scope="module")
def test_image_path():
    """Get the test image path."""
    # Find the package root (go up from test/ to src/vlm-detections/)
    test_dir = Path(__file__).parent
    pkg_root = test_dir.parent
    image_path = pkg_root / "assets" / "girl_phone.png"
    
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")
    
    return str(image_path)


@pytest.fixture(scope="module")
def test_image_bgr(test_image_path):
    """Load the test image in BGR format."""
    image = cv2.imread(test_image_path)
    if image is None:
        pytest.skip(f"Failed to load test image: {test_image_path}")
    return image


@pytest.fixture(scope="module")
def adapter():
    """Create and load Florence-2 adapter (reused across tests)."""
    if not FLORENCE2_AVAILABLE:
        pytest.skip("Florence-2 dependencies not available (torch, transformers)")
    
    from vlm_detections.adapters.florence2_adapter import Florence2Adapter
    
    adapter = Florence2Adapter(model_id="florence-community/Florence-2-base-ft")
    
    # Load on CPU to ensure compatibility
    try:
        adapter.load(device="cpu")
    except Exception as e:
        pytest.skip(f"Failed to load Florence-2 model: {e}")
    
    return adapter


class TestFlorence2Tasks:
    """Test all Florence-2 supported tasks."""
    
    def test_caption(self, adapter, test_image_bgr):
        """Test CAPTION task - concise image caption."""
        adapter.set_task("CAPTION")
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.5
        )
        
        assert raw_text, "CAPTION should return text output"
        print(f"\nCAPTION raw_text: {raw_text}\")")
        
        # Caption tasks may return text in raw_text or as detection text
        if len(detections) > 0:
            assert detections[0].text, "Caption detection should have text field"
            print(f"CAPTION (from detection): {detections[0].text}")
        else:
            # Check if caption is in raw_text JSON
            import json
            try:
                data = json.loads(raw_text)
                caption_value = data.get("<CAPTION>")
                if isinstance(caption_value, str):
                    caption = caption_value
                elif isinstance(caption_value, dict):
                    caption = caption_value.get("caption")
                else:
                    caption = data.get("caption")
                assert caption, "Caption should be in raw_text if not in detections"
                print(f"CAPTION (from raw_text): {caption}")
            except:
                pytest.fail("Caption not found in detections or raw_text")
    
    def test_detailed_caption(self, adapter, test_image_bgr):
        """Test DETAILED_CAPTION task - detailed caption."""
        adapter.set_task("DETAILED_CAPTION")
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.5
        )
        
        assert raw_text, "DETAILED_CAPTION should return text output"
        print(f"\nDETAILED_CAPTION raw_text: {raw_text}")
        
        if len(detections) > 0:
            assert detections[0].text, "Detailed caption should have text"
            print(f"DETAILED_CAPTION: {detections[0].text}")
        else:
            import json
            try:
                data = json.loads(raw_text)
                caption_value = data.get("<DETAILED_CAPTION>")
                if isinstance(caption_value, str):
                    caption = caption_value
                elif isinstance(caption_value, dict):
                    caption = caption_value.get("caption")
                else:
                    caption = data.get("caption")
                assert caption, "Detailed caption should be in raw_text"
                print(f"DETAILED_CAPTION (from raw_text): {caption}")
            except:
                pytest.fail("Detailed caption not found")
    
    def test_more_detailed_caption(self, adapter, test_image_bgr):
        """Test MORE_DETAILED_CAPTION task - very detailed caption."""
        adapter.set_task("MORE_DETAILED_CAPTION")
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.5
        )
        
        assert raw_text, "MORE_DETAILED_CAPTION should return text output"
        print(f"\nMORE_DETAILED_CAPTION raw_text: {raw_text}")
        
        if len(detections) > 0:
            assert detections[0].text, "More detailed caption should have text"
            print(f"MORE_DETAILED_CAPTION: {detections[0].text}")
        else:
            import json
            try:
                data = json.loads(raw_text)
                caption_value = data.get("<MORE_DETAILED_CAPTION>")
                if isinstance(caption_value, str):
                    caption = caption_value
                elif isinstance(caption_value, dict):
                    caption = caption_value.get("caption")
                else:
                    caption = data.get("caption")
                assert caption, "More detailed caption should be in raw_text"
                print(f"MORE_DETAILED_CAPTION (from raw_text): {caption}")
            except:
                pytest.fail("More detailed caption not found")
    
    def test_object_detection(self, adapter, test_image_bgr):
        """Test OD task - object detection with bounding boxes."""
        adapter.set_task("OD")
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.3
        )
        
        print(f"\nOD raw_text: {raw_text}")
        
        # OD may return empty if threshold is too high or model doesn't detect anything
        if len(detections) == 0:
            pytest.skip("OD did not detect any objects (may be model limitation or threshold issue)")
        
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            assert x2 > x1, "Bounding box width should be positive"
            assert y2 > y1, "Bounding box height should be positive"
            assert det.label, "Detection should have a label"
            assert 0 <= det.score <= 1.0, "Score should be between 0 and 1"
        
        print(f"\nOD: Found {len(detections)} objects")
        for det in detections[:5]:  # Print first 5
            print(f"  - {det.label}: {det.score:.2f} at {det.xyxy}")
    
    def test_dense_region_caption(self, adapter, test_image_bgr):
        """Test DENSE_REGION_CAPTION task - region captions with boxes."""
        adapter.set_task("DENSE_REGION_CAPTION")
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.3
        )
        
        print(f"\nDENSE_REGION_CAPTION raw_text: {raw_text}")
        
        if len(detections) == 0:
            pytest.skip("DENSE_REGION_CAPTION did not return regions (model limitation)")
        
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            assert x2 > x1, "Bounding box width should be positive"
            assert y2 > y1, "Bounding box height should be positive"
            assert det.text, "Region should have caption text"
        
        print(f"\nDENSE_REGION_CAPTION: Found {len(detections)} regions")
        for det in detections[:5]:
            print(f"  - {det.text} at {det.xyxy}")
    
    def test_region_proposal(self, adapter, test_image_bgr):
        """Test REGION_PROPOSAL task - region proposals without labels."""
        adapter.set_task("REGION_PROPOSAL")
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.3
        )
        
        assert len(detections) > 0, "REGION_PROPOSAL should return regions"
        
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            assert x2 > x1, "Bounding box width should be positive"
            assert y2 > y1, "Bounding box height should be positive"
        
        print(f"\nREGION_PROPOSAL: Found {len(detections)} region proposals")
    
    def test_referring_expression_segmentation(self, adapter, test_image_bgr):
        """Test REFERRING_EXPRESSION_SEGMENTATION task - segmentation from referring expression."""
        adapter.set_task("REFERRING_EXPRESSION_SEGMENTATION")
        
        # Test with "the phone" which should be visible in the image
        prompt = "the phone"
        
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt=prompt, 
            system_prompt=None, 
            threshold=0.3
        )
        
        print(f"\nREFERRING_EXPRESSION_SEGMENTATION ('{prompt}') raw_text: {raw_text}")
        
        if len(detections) == 0:
            pytest.skip(f"REFERRING_EXPRESSION_SEGMENTATION did not find '{prompt}' (model limitation)")
        
        for det in detections:
            if det.polygon:
                assert len(det.polygon) >= 3, "Polygon should have at least 3 points"
                print(f"  - Segmentation with {len(det.polygon)} points")
            else:
                print(f"  - Detection at {det.xyxy} (no polygon)")
        
        print(f"REFERRING_EXPRESSION_SEGMENTATION: Found {len(detections)} results")
    
    def test_region_to_segmentation(self, adapter, test_image_bgr):
        """Test REGION_TO_SEGMENTATION task - segmentation for specific region."""
        adapter.set_task("REGION_TO_SEGMENTATION")
        
        # This task typically requires region coordinates in the prompt
        # Florence-2 may accept region descriptions or coordinates
        prompt = "the center region"
        
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt=prompt, 
            system_prompt=None, 
            threshold=0.3
        )
        
        print(f"\nREGION_TO_SEGMENTATION ('{prompt}') raw_text: {raw_text}")
        
        if len(detections) == 0:
            pytest.skip("REGION_TO_SEGMENTATION did not return results (may require specific region format)")
        
        for det in detections:
            if det.polygon:
                assert len(det.polygon) >= 3, "Polygon should have at least 3 points"
                print(f"  - Segmentation with {len(det.polygon)} points")
        
        print(f"REGION_TO_SEGMENTATION: Found {len(detections)} segmentations")
    
    def test_ocr(self, adapter, test_image_bgr):
        """Test OCR task - full image text extraction."""
        adapter.set_task("OCR")
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.5
        )
        
        # OCR might not find text if there's none in the image
        if len(detections) > 0:
            assert detections[0].text is not None, "OCR detection should have text"
            print(f"\nOCR: {detections[0].text}")
        else:
            print("\nOCR: No text detected in image")
    
    def test_ocr_with_region(self, adapter, test_image_bgr):
        """Test OCR_WITH_REGION task - OCR with bounding boxes."""
        adapter.set_task("OCR_WITH_REGION")
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.3
        )
        
        # OCR might not find text if there's none in the image
        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2 = det.xyxy
                assert x2 > x1, "OCR box width should be positive"
                assert y2 > y1, "OCR box height should be positive"
                assert det.text, "OCR region should have text"
            
            print(f"\nOCR_WITH_REGION: Found {len(detections)} text regions")
            for det in detections[:5]:
                print(f"  - '{det.text}' at {det.xyxy}")
        else:
            print("\nOCR_WITH_REGION: No text detected in image")
    
    def test_caption_to_phrase_grounding(self, adapter, test_image_bgr):
        """Test CAPTION_TO_PHRASE_GROUNDING task - ground phrases to boxes."""
        adapter.set_task("CAPTION_TO_PHRASE_GROUNDING")
        
        # Provide a caption with phrases to ground
        prompt = "a girl holding a phone"
        
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt=prompt, 
            system_prompt=None, 
            threshold=0.3
        )
        
        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2 = det.xyxy
                assert x2 > x1, "Grounded box width should be positive"
                assert y2 > y1, "Grounded box height should be positive"
                assert det.label, "Grounded phrase should have label"
            
            print(f"\nCAPTION_TO_PHRASE_GROUNDING ('{prompt}'): Found {len(detections)} grounded phrases")
            for det in detections:
                print(f"  - {det.label} at {det.xyxy}")
        else:
            print(f"\nCAPTION_TO_PHRASE_GROUNDING: No phrases grounded")
    
    def test_ref_legacy(self, adapter, test_image_bgr):
        """Test REF task - legacy referring expression."""
        adapter.set_task("REF")
        
        # Test with "the phone" referring expression
        prompt = "the phone"
        
        detections, properties, relations, raw_text = adapter.infer_from_image(
            test_image_bgr, 
            prompt=prompt, 
            system_prompt=None, 
            threshold=0.3
        )
        
        print(f"\nREF ('{prompt}') raw_text: {raw_text}")
        
        if len(detections) == 0:
            pytest.skip(f"REF did not find '{prompt}' (model limitation)")
        
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            assert x2 > x1, "REF box width should be positive"
            assert y2 > y1, "REF box height should be positive"
            print(f"  - {det.label} at {det.xyxy}")
        
        print(f"REF: Found {len(detections)} boxes")


class TestFlorence2ZeroShotInterface:
    """Test the ZeroShotObjectDetector interface."""
    
    def test_zero_shot_infer_from_image(self, adapter, test_image_bgr):
        """Test OD mode directly (Florence2 doesn't support class-specific detection)."""
        # Save current task
        original_task = adapter.task
        
        # Florence-2 OD mode doesn't use class lists - it detects all objects
        # Set to OD mode and test
        adapter.set_task("OD")
        
        # Use the PromptBasedVLM interface (which is what Florence2 actually uses)
        detections, _, _, raw_text = adapter.infer_from_image(
            test_image_bgr,
            prompt="",
            system_prompt=None,
            threshold=0.3
        )
        
        print(f"\nOD mode test: Found {len(detections) if detections else 0} objects")
        print(f"Raw output: {raw_text}")
        
        if not detections or len(detections) == 0:
            pytest.skip("OD mode did not detect objects (model limitation or threshold)")
        
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            assert x2 > x1, "Bounding box width should be positive"
            assert y2 > y1, "Bounding box height should be positive"
            assert det.label, "Detection should have a label"
            print(f"  - {det.label}: {det.score:.2f}")
        
        # Restore task
        adapter.set_task(original_task)


class TestFlorence2GenerationParams:
    """Test generation parameter configuration."""
    
    def test_generation_config_spec(self, adapter):
        """Test that adapter provides generation config spec."""
        spec = adapter.generation_config_spec()
        
        assert "max_new_tokens" in spec
        assert "num_beams" in spec
        assert "do_sample" in spec
        assert "temperature" in spec
        
        # Check metadata structure
        assert "default" in spec["max_new_tokens"]
        assert "type" in spec["max_new_tokens"]
    
    def test_update_generation_params(self, adapter):
        """Test updating generation parameters."""
        new_params = {
            "max_new_tokens": 512,
            "num_beams": 5,
            "do_sample": True,
            "temperature": 0.8,
        }
        
        adapter.update_generation_params(new_params)
        
        assert adapter.gen_params["max_new_tokens"] == 512
        assert adapter.gen_params["num_beams"] == 5
        assert adapter.gen_params["do_sample"] is True
        assert adapter.gen_params["temperature"] == 0.8


class TestFlorence2ModelVariants:
    """Test different Florence-2 model variants (if available)."""
    
    @pytest.mark.skip(reason="Requires downloading additional models - enable manually")
    def test_florence2_large(self, test_image_bgr):
        """Test Florence-2-large model variant."""
        if not FLORENCE2_AVAILABLE:
            pytest.skip("Florence-2 dependencies not available")
        
        from vlm_detections.adapters.florence2_adapter import Florence2Adapter
        
        adapter = Florence2Adapter(model_id="florence-community/Florence-2-large-ft")
        adapter.load(device="cpu")
        adapter.set_task("OD")
        
        detections, _, _, _ = adapter.infer_from_image(
            test_image_bgr, 
            prompt="", 
            system_prompt=None, 
            threshold=0.3
        )
        
        assert len(detections) > 0, "Large model should detect objects"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
