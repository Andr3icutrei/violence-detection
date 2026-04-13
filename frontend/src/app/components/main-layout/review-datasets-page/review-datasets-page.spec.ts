import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ReviewDatasetsPage } from './review-datasets-page';

describe('ReviewDatasetsPage', () => {
  let component: ReviewDatasetsPage;
  let fixture: ComponentFixture<ReviewDatasetsPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ReviewDatasetsPage],
    }).compileComponents();

    fixture = TestBed.createComponent(ReviewDatasetsPage);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
