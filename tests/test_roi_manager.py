"""
Tests for ROI manager functionality
"""
import pytest
import numpy as np
from visionsub.core.roi_manager import ROIManager, ROIType, ROIInfo


class TestROIManager:
    """Test ROI manager functionality"""
    
    def test_roi_manager_initialization(self):
        """Test ROI manager initialization"""
        manager = ROIManager()
        assert len(manager.rois) > 0  # Should have default ROIs
        assert manager.active_roi_id is not None
        assert manager.roi_counter > 0
    
    def test_add_roi(self):
        """Test adding ROI"""
        manager = ROIManager()
        initial_count = len(manager.rois)
        
        roi_id = manager.add_roi(
            name="Test ROI",
            roi_type=ROIType.CUSTOM,
            rect=(100, 200, 300, 400),
            description="Test ROI description",
            confidence_threshold=0.8,
            language="zh"
        )
        
        assert roi_id in manager.rois
        assert len(manager.rois) == initial_count + 1
        assert manager.rois[roi_id].name == "Test ROI"
        assert manager.rois[roi_id].type == ROIType.CUSTOM
        assert manager.rois[roi_id].rect == (100, 200, 300, 400)
        assert manager.rois[roi_id].confidence_threshold == 0.8
        assert manager.rois[roi_id].language == "zh"
    
    def test_remove_roi(self):
        """Test removing ROI"""
        manager = ROIManager()
        initial_count = len(manager.rois)
        
        # Add a test ROI
        roi_id = manager.add_roi(
            name="Test ROI",
            roi_type=ROIType.CUSTOM,
            rect=(100, 200, 300, 400)
        )
        
        assert len(manager.rois) == initial_count + 1
        
        # Remove the ROI
        success = manager.remove_roi(roi_id)
        assert success is True
        assert roi_id not in manager.rois
        assert len(manager.rois) == initial_count
    
    def test_update_roi(self):
        """Test updating ROI"""
        manager = ROIManager()
        
        # Add a test ROI
        roi_id = manager.add_roi(
            name="Test ROI",
            roi_type=ROIType.CUSTOM,
            rect=(100, 200, 300, 400)
        )
        
        # Update the ROI
        success = manager.update_roi(
            roi_id,
            name="Updated ROI",
            rect=(200, 300, 400, 500),
            confidence_threshold=0.9
        )
        
        assert success is True
        assert manager.rois[roi_id].name == "Updated ROI"
        assert manager.rois[roi_id].rect == (200, 300, 400, 500)
        assert manager.rois[roi_id].confidence_threshold == 0.9
    
    def test_set_roi_enabled(self):
        """Test setting ROI enabled state"""
        manager = ROIManager()
        
        # Add a test ROI
        roi_id = manager.add_roi(
            name="Test ROI",
            roi_type=ROIType.CUSTOM,
            rect=(100, 200, 300, 400)
        )
        
        # Initially enabled
        assert manager.rois[roi_id].enabled is True
        
        # Disable ROI
        success = manager.set_roi_enabled(roi_id, False)
        assert success is True
        assert manager.rois[roi_id].enabled is False
        
        # Enable ROI
        success = manager.set_roi_enabled(roi_id, True)
        assert success is True
        assert manager.rois[roi_id].enabled is True
    
    def test_active_roi_management(self):
        """Test active ROI management"""
        manager = ROIManager()
        
        # Add a test ROI
        roi_id = manager.add_roi(
            name="Test ROI",
            roi_type=ROIType.CUSTOM,
            rect=(100, 200, 300, 400)
        )
        
        # Set as active ROI
        success = manager.set_active_roi(roi_id)
        assert success is True
        assert manager.active_roi_id == roi_id
        assert manager.get_active_roi().id == roi_id
    
    def test_get_roi_methods(self):
        """Test ROI retrieval methods"""
        manager = ROIManager()
        
        # Add test ROIs
        roi1_id = manager.add_roi("ROI 1", ROIType.CUSTOM, (100, 200, 300, 400))
        roi2_id = manager.add_roi("ROI 2", ROIType.CUSTOM, (200, 300, 400, 500))
        
        # Disable one ROI
        manager.set_roi_enabled(roi2_id, False)
        
        # Test get_all_rois
        all_rois = manager.get_all_rois()
        assert len(all_rois) == len(manager.rois)
        
        # Test get_enabled_rois
        enabled_rois = manager.get_enabled_rois()
        assert len(enabled_rois) < len(all_rois)
        assert all(roi.enabled for roi in enabled_rois)
        
        # Test get_roi
        roi = manager.get_roi(roi1_id)
        assert roi is not None
        assert roi.id == roi1_id
        
        # Test get_nonexistent_roi
        nonexistent = manager.get_roi("nonexistent")
        assert nonexistent is None
    
    def test_apply_roi_to_frame(self):
        """Test applying ROI to frame"""
        manager = ROIManager()
        
        # Create test frame (480x640)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add test ROI
        roi_id = manager.add_roi(
            name="Test ROI",
            roi_type=ROIType.CUSTOM,
            rect=(100, 100, 200, 150)  # x=100, y=100, w=200, h=150
        )
        
        # Set as active ROI
        manager.set_active_roi(roi_id)
        
        # Apply ROI to frame
        roi_frame = manager.apply_roi_to_frame(frame)
        
        # Check ROI frame dimensions
        assert roi_frame.shape == (150, 200, 3)  # height=150, width=200
        
        # Test with disabled ROI
        manager.set_roi_enabled(roi_id, False)
        disabled_frame = manager.apply_roi_to_frame(frame)
        assert disabled_frame.shape == frame.shape  # Should return original frame
        
        # Test with invalid ROI dimensions
        manager.update_roi(roi_id, rect=(-10, -10, -1, -1))
        invalid_frame = manager.apply_roi_to_frame(frame)
        assert invalid_frame.shape == frame.shape  # Should return original frame
    
    def test_roi_config_serialization(self):
        """Test ROI configuration serialization"""
        manager = ROIManager()
        
        # Add test ROI
        roi_id = manager.add_roi(
            name="Test ROI",
            roi_type=ROIType.CUSTOM,
            rect=(100, 200, 300, 400),
            description="Test description",
            confidence_threshold=0.8,
            language="zh"
        )
        
        # Get ROI config
        config = manager.get_roi_config(roi_id)
        
        assert config["roi_rect"] == (100, 200, 300, 400)
        assert config["roi_enabled"] is True
        assert config["roi_type"] == "custom"
        assert config["roi_id"] == roi_id
        assert config["confidence_threshold"] == 0.8
        assert config["language"] == "zh"
    
    def test_roi_preset_import_export(self):
        """Test ROI preset import and export"""
        manager = ROIManager()
        
        # Test import
        presets = [
            {
                "name": "Imported ROI 1",
                "type": "custom",
                "rect": [50, 60, 200, 150],
                "description": "Imported test ROI",
                "confidence_threshold": 0.7,
                "language": "en"
            },
            {
                "name": "Imported ROI 2",
                "type": "subtitle",
                "rect": [0, 400, 640, 80],
                "description": "Imported subtitle ROI"
            }
        ]
        
        imported_count = manager.import_roi_presets(presets)
        assert imported_count == 2
        
        # Test export
        exported_presets = manager.export_roi_presets()
        assert len(exported_presets) >= 2
        
        # Verify imported ROIs exist
        roi_names = [roi.name for roi in manager.get_all_rois()]
        assert "Imported ROI 1" in roi_names
        assert "Imported ROI 2" in roi_names
    
    def test_roi_scaling(self):
        """Test ROI scaling functionality"""
        manager = ROIManager()
        
        # Test scaling from 1920x1080 to 1280x720
        original_rect = (0, 900, 1920, 180)  # Bottom subtitle area
        original_res = (1920, 1080)
        target_res = (1280, 720)
        
        scaled_rect = manager.scale_roi_to_resolution(
            original_rect, original_res, target_res
        )
        
        # Calculate expected values
        scale_x = 1280 / 1920
        scale_y = 720 / 1080
        expected_x = int(0 * scale_x)
        expected_y = int(900 * scale_y)
        expected_w = int(1920 * scale_x)
        expected_h = int(180 * scale_y)
        
        assert scaled_rect == (expected_x, expected_y, expected_w, expected_h)
    
    def test_roi_statistics(self):
        """Test ROI statistics"""
        manager = ROIManager()
        
        # Add some test ROIs
        manager.add_roi("Test Custom", ROIType.CUSTOM, (100, 100, 200, 150))
        manager.add_roi("Test Subtitle", ROIType.SUBTITLE, (0, 400, 640, 80))
        
        # Disable one ROI
        all_rois = manager.get_all_rois()
        if all_rois:
            manager.set_roi_enabled(all_rois[0].id, False)
        
        stats = manager.get_roi_statistics()
        
        assert "total_rois" in stats
        assert "enabled_rois" in stats
        assert "disabled_rois" in stats
        assert "active_roi_id" in stats
        assert "type_distribution" in stats
        
        assert stats["total_rois"] == len(all_rois)
        assert stats["enabled_rois"] + stats["disabled_rois"] == stats["total_rois"]
        assert stats["active_roi_id"] == manager.active_roi_id
        assert isinstance(stats["type_distribution"], dict)
    
    def test_roi_clear_all(self):
        """Test clearing all ROIs"""
        manager = ROIManager()
        
        # Add some test ROIs
        manager.add_roi("Test ROI 1", ROIType.CUSTOM, (100, 100, 200, 150))
        manager.add_roi("Test ROI 2", ROIType.SUBTITLE, (0, 400, 640, 80))
        
        initial_count = len(manager.rois)
        assert initial_count > 0
        
        # Clear all ROIs
        manager.clear_all_rois()
        
        # Should have default ROIs again
        assert len(manager.rois) > 0
        assert manager.roi_counter > 0
        
        # Should have default ROIs like "全屏"
        roi_names = [roi.name for roi in manager.get_all_rois()]
        assert "全屏" in roi_names