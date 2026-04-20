import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ReviewDatasetItem } from './review-dataset-item';

describe('ReviewDatasetItem', () => {
  let component: ReviewDatasetItem;
  let fixture: ComponentFixture<ReviewDatasetItem>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ReviewDatasetItem],
    }).compileComponents();

    fixture = TestBed.createComponent(ReviewDatasetItem);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
